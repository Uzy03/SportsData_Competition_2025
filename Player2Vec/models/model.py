import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForCausalLM

class PatchEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        # 空間的特徴を学習するための1D畳み込み層
        # 入力: (batch, 23, time, 2) → (batch, 23, time, hidden_dim)
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 時系列特徴を学習するためのLSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 最終的な特徴抽出
        self.fc = nn.Linear(hidden_dim * 2, 4096)  # bidirectionalなので*2
        
        # ドロップアウト
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 入力: x.shape = (batch, 23, time, 2)
        batch_size, num_players, time_steps, coords = x.shape
        
        # 各選手の時系列データを処理
        player_features = []
        for i in range(num_players):
            # 選手iの時系列データ: (batch, time, 2)
            player_data = x[:, i, :, :]  # (batch, time, 2)
            
            # 空間的特徴の抽出
            spatial_feat = self.spatial_conv(player_data.transpose(1, 2))  # (B, hidden, T)
            spatial_feat = spatial_feat.transpose(1, 2)                    # (B, T, hidden)
            
            # 時系列特徴の抽出
            _, (hidden, _) = self.lstm(spatial_feat)
            if self.lstm.bidirectional:
                final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                final_hidden = hidden[-1]
            player_features.append(final_hidden)
        
        # (batch, num_players, hidden_dim * 2)
        combined_features = torch.stack(player_features, dim=1)
        
        # シンプルattention: 平均
        attended_features = torch.mean(combined_features, dim=1)
        
        output = self.fc(self.dropout(attended_features))
        return output

class MultiTaskHeads(nn.Module):
    def __init__(self, emb_dim=4096):
        super().__init__()
        self.delta_head = nn.Linear(emb_dim, 23 * 18)  # 23選手 × 18次元（9フレーム × 2座標）

    def forward(self, x):
        delta = self.delta_head(x).view(x.size(0), 23, 18)  # (B, 23, 18)
        return delta

class SLMWrapper(nn.Module):
    def __init__(self, model_name: str = "distilgpt2"):
        super().__init__()
        # Force slow tokenizer to avoid protobuf requirement when converting SP/T5 tokenizers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        # Ensure pad token is set to eos if missing (common for GPT2 family)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        for p in self.model.parameters():
            p.requires_grad = False
        self.hidden = self.model.config.hidden_size
        self.p_proj = nn.Linear(256, self.hidden)
        self.prefix_ln = nn.LayerNorm(self.hidden)

    def _ensure_special_token(self, token: str) -> int:
        """Ensure a special token exists in tokenizer and model embeddings.
        Returns its token id.
        """
        vocab = self.tokenizer.get_vocab()
        if token in vocab:
            return vocab[token]
        # add and resize
        self.tokenizer.add_special_tokens({"additional_special_tokens": [token]})
        self.model.resize_token_embeddings(len(self.tokenizer))
        # fetch id after resize
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        return token_id

    @torch.no_grad()
    def style_probe(
        self,
        p: torch.Tensor,
        prompt: str,
        *,
        style_token: str = "<STYLE>",
        prefix_scale: float = 0.05,
        prefix_len: int = 8,
        no_prefix: bool = False,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Return the last-layer hidden state at the style_token position.
        - Inject soft prefix derived from p (same as generate)
        - Append/ensure style_token is in the prompt
        Returns: Tensor(hidden)
        """
        device = next(self.parameters()).device
        if p.dim() == 1:
            p = p.unsqueeze(0)
        # Ensure special token exists and in the prompt
        style_id = self._ensure_special_token(style_token)
        if style_token not in prompt:
            prompt = prompt + " " + style_token
        tokens = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = tokens.input_ids.to(device)
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        # Optional seed
        if seed is not None:
            try:
                torch.manual_seed(seed)
            except Exception:
                pass
        # prefix
        if not no_prefix:
            prefix_vec = self.prefix_ln(self.p_proj(p))  # (B, hidden)
            prefix_vec = prefix_scale * prefix_vec
            soft_prefix = prefix_vec.unsqueeze(1).expand(-1, int(max(prefix_len, 1)), -1)
            inputs_embeds = torch.cat([soft_prefix, inputs_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.size()[:2], dtype=torch.long, device=device)
        # forward with hidden states
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # For CausalLM, last hidden is in hidden_states[-1]
        if getattr(outputs, "hidden_states", None) is None or len(outputs.hidden_states) == 0:
            raise RuntimeError("Model did not return hidden_states; ensure output_hidden_states=True and model supports it")
        last_hidden = outputs.hidden_states[-1]  # (B, T, H)
        # locate style token position within input_ids (before prefix), then offset by prefix_len
        style_positions = (input_ids[0] == style_id).nonzero(as_tuple=False)
        if style_positions.numel() == 0:
            # fallback: use last token position (before prefix)
            pos = input_ids.size(1) - 1
        else:
            pos = int(style_positions[-1].item())
        if not no_prefix:
            pos = pos + int(max(prefix_len, 1))
        vec = last_hidden[0, pos, :].detach()
        return vec

    def generate(
        self,
        p: torch.Tensor,
        question: str,
        max_new_tokens: int = 32,
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 3,
        prefix_scale: float = 0.05,
        prefix_len: int = 8,
        no_prefix: bool = False,
        # extras
        length_penalty: float = 1.0,
        num_beams: int = 1,
        do_sample: bool = True,
        seed: int | None = None,
    ) -> str:
        device = next(self.parameters()).device
        if p.dim() == 1:
            p = p.unsqueeze(0)
        tokens = self.tokenizer(question, return_tensors="pt", add_special_tokens=False)
        input_ids = tokens.input_ids.to(device)
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        # Optional seeding for reproducibility
        if seed is not None:
            try:
                torch.manual_seed(seed)
            except Exception:
                pass
        if not no_prefix:
            # LayerNorm then scale; repeat across prefix_len
            prefix_vec = self.prefix_ln(self.p_proj(p))  # (B, hidden)
            prefix_vec = prefix_scale * prefix_vec
            soft_prefix = prefix_vec.unsqueeze(1).expand(-1, int(max(prefix_len, 1)), -1)  # (B,P,hidden)
            inputs_embeds = torch.cat([soft_prefix, inputs_embeds], dim=1)
        # Build attention mask for prefix+tokens
        attention_mask = torch.ones(inputs_embeds.size()[:2], dtype=torch.long, device=device)
        # Input length guard (trim to model max positions)
        max_pos = getattr(self.model.config, "max_position_embeddings", None)
        if isinstance(max_pos, int) and inputs_embeds.size(1) > max_pos:
            # Keep the last max_pos tokens (prefixは先頭にあるため、全体長でスライス)
            inputs_embeds = inputs_embeds[:, -max_pos:, :]
            attention_mask = attention_mask[:, -max_pos:]
        output_ids = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=do_sample if num_beams <= 1 else False,
            temperature=temperature,
            top_p=top_p,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_beams=num_beams,
        )
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return text

class SoccerLightningModule(pl.LightningModule):
    def __init__(self, num_players=100, num_teams=2, lr: float = 1e-3,
                 model_name: str = "distilgpt2",
                 prefix_scale: float = 0.05,
                 prefix_len: int = 8,
                 no_prefix: bool = False,
                 style_token: str = "<STYLE>",
                 prompt: str | None = None):
        super().__init__()
        self.save_hyperparameters()
        self.patch_encoder = PatchEncoder()
        # Frozen SLM wrapper
        self.slm = SLMWrapper(model_name=model_name)
        for p in self.slm.parameters():
            p.requires_grad = False
        # Reduce PatchEncoder(4096) -> 256 for SLM p_proj input
        self.p_reduce = nn.Linear(4096, 256)
        # Heads expect SLM hidden size
        self.heads = MultiTaskHeads(emb_dim=self.slm.hidden)
        # 自己教師あり: 軌道予測のMSE
        self.criterion_delta = nn.MSELoss(reduction='none')
        # コントラスト学習
        self.tau = 0.07
        self.lr = lr
        # ロス重み
        self.lambda_delta = 1.0
        self.lambda_infonce = 0.5
        # SLM config
        self.prefix_scale = prefix_scale
        self.prefix_len = prefix_len
        self.no_prefix = no_prefix
        self.style_token = style_token
        if prompt is None:
            self.prompt = (
                "[INST] あなたはサッカーアナリストです。\n"
                "以下の情報を読み取り、直後の <STYLE> の内部表現に\n"
                "選手のプレースタイルを要約して保持してください。\n"
                "<STYLE> その後のテキストは無視して構いません。 [/INST]"
            )
        else:
            self.prompt = prompt

    def forward(self, patch):
        # Returns normalized SLM-derived embedding for a batch of patches
        return self._slm_embed_from_patch(patch)

    def _slm_embed_from_patch(self, patch: torch.Tensor) -> torch.Tensor:
        """Compute SLM hidden vectors from PatchEncoder outputs.
        1) PatchEncoder -> 4096-d
        2) reduce to 256-d
        3) inject as soft prefix into frozen SLM and take hidden at style token
        4) L2 normalize
        """
        device = patch.device
        # (B, 4096)
        p4096 = F.normalize(self.patch_encoder(patch), dim=-1)
        # (B, 256)
        p256 = self.p_reduce(p4096)
        # Per-sample style probe (style_probe currently handles single prompt sequence)
        vecs = []
        for i in range(p256.size(0)):
            v = self.slm.style_probe(
                p256[i],
                self.prompt,
                style_token=self.style_token,
                prefix_scale=self.prefix_scale,
                prefix_len=self.prefix_len,
                no_prefix=self.no_prefix,
            )
            vecs.append(v)
        emb = torch.stack(vecs, dim=0).to(device)
        emb = F.normalize(emb, dim=-1)
        return emb

    @staticmethod
    def _last_step_delta(patch: torch.Tensor) -> torch.Tensor:
        """(B, 23, T, 2) → 最後の10フレームの軌道 (B, 23, 20)"""
        # T>=10を仮定。足りなければゼロ
        if patch.size(2) < 10:
            return torch.zeros(patch.size(0), 23, 20, device=patch.device, dtype=patch.dtype)
        
        # 最後の10フレームを取得
        last_10_frames = patch[:, :, -10:, :]  # (B, 23, 10, 2)
        
        # 各フレーム間の差分を計算（9個の差分）
        deltas = []
        for i in range(9):
            current = last_10_frames[:, :, i+1, :]  # 現在のフレーム
            previous = last_10_frames[:, :, i, :]   # 前のフレーム
            delta = current - previous               # 差分
            deltas.append(delta)
        
        # 9個の差分を結合 (B, 23, 9, 2) → (B, 23, 18)
        deltas_tensor = torch.stack(deltas, dim=2)  # (B, 23, 9, 2)
        deltas_flat = deltas_tensor.view(deltas_tensor.size(0), deltas_tensor.size(1), -1)  # (B, 23, 18)
        
        return deltas_flat

    def _masked_mse(self, pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # pred/target: (B, 23, 2), valid_mask: (B, 23)
        loss = (pred - target) ** 2  # (B, 23, 2)
        loss = loss.mean(dim=-1)     # (B, 23)
        loss = loss * valid_mask
        denom = valid_mask.sum().clamp_min(1.0)
        return loss.sum() / denom

    def training_step(self, batch, batch_idx):
        patch, player_ids, team_id = batch  # player_ids, team_idは未使用
        # Use SLM-derived embeddings
        emb = self._slm_embed_from_patch(patch)
        # 軌道予測（最後の2フレームのdelta）
        pred_delta = self.heads(emb)
        target_delta = self._last_step_delta(patch)
        # マスク: 最後の2フレームが両方ゼロのエンティティは無効
        last = patch[:, :, -1, :]
        prev = patch[:, :, -2, :] if patch.size(2) >= 2 else torch.zeros_like(last)
        valid_mask = (((last.abs().sum(dim=-1) + prev.abs().sum(dim=-1)) > 0).float())
        loss_delta = self._masked_mse(pred_delta, target_delta, valid_mask)
        # InfoNCE
        sim = emb @ emb.t()
        sim = sim / self.tau
        targets = torch.arange(sim.size(0), device=sim.device)
        loss_infonce = F.cross_entropy(sim, targets)
        # 合成
        loss = self.lambda_delta * loss_delta + self.lambda_infonce * loss_infonce
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_loss_delta", loss_delta)
        self.log("train_loss_infonce", loss_infonce)
        return loss

    def validation_step(self, batch, batch_idx):
        patch, player_ids, team_id = batch
        emb = self._slm_embed_from_patch(patch)
        pred_delta = self.heads(emb)
        target_delta = self._last_step_delta(patch)
        last = patch[:, :, -1, :]
        prev = patch[:, :, -2, :] if patch.size(2) >= 2 else torch.zeros_like(last)
        valid_mask = (((last.abs().sum(dim=-1) + prev.abs().sum(dim=-1)) > 0).float())
        loss_delta = self._masked_mse(pred_delta, target_delta, valid_mask)
        sim = emb @ emb.t()
        sim = sim / self.tau
        targets = torch.arange(sim.size(0), device=sim.device)
        loss_infonce = F.cross_entropy(sim, targets)
        loss = self.lambda_delta * loss_delta + self.lambda_infonce * loss_infonce
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_loss_delta", loss_delta)
        self.log("val_loss_infonce", loss_infonce)
        return loss

    def test_step(self, batch, batch_idx):
        patch, player_ids, team_id = batch
        emb = self._slm_embed_from_patch(patch)
        pred_delta = self.heads(emb)
        target_delta = self._last_step_delta(patch)
        last = patch[:, :, -1, :]
        prev = patch[:, :, -2, :] if patch.size(2) >= 2 else torch.zeros_like(last)
        valid_mask = (((last.abs().sum(dim=-1) + prev.abs().sum(dim=-1)) > 0).float())
        loss_delta = self._masked_mse(pred_delta, target_delta, valid_mask)
        sim = emb @ emb.t()
        sim = sim / self.tau
        targets = torch.arange(sim.size(0), device=sim.device)
        loss_infonce = F.cross_entropy(sim, targets)
        loss = self.lambda_delta * loss_delta + self.lambda_infonce * loss_infonce
        self.log("test_loss", loss)
        self.log("test_loss_delta", loss_delta)
        self.log("test_loss_infonce", loss_infonce)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)