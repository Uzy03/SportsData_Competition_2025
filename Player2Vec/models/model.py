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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        # Ensure pad token is set to eos if missing (common for GPT2 family)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        for p in self.model.parameters():
            p.requires_grad = False
        self.hidden = self.model.config.hidden_size
        self.encoder_prefix = nn.Parameter(torch.zeros(32, self.hidden))
        self.decoder_prefix = nn.Parameter(torch.zeros(1, self.hidden))
        self.p_proj = nn.Linear(256, self.hidden)

    def generate(self, p: torch.Tensor, question: str, max_new_tokens: int = 32) -> str:
        device = next(self.parameters()).device
        if p.dim() == 1:
            p = p.unsqueeze(0)
        prefix = self.p_proj(p) + self.decoder_prefix  # (B, hidden)
        tokens = self.tokenizer(question, return_tensors="pt", add_special_tokens=False)
        input_ids = tokens.input_ids.to(device)
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prefix.unsqueeze(1), inputs_embeds], dim=1)
        # Build attention mask for prefix+tokens
        attention_mask = torch.ones(inputs_embeds.size()[:2], dtype=torch.long, device=device)
        output_ids = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            temperature=0.7,
            top_p=0.9,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
        )
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return text

class SoccerLightningModule(pl.LightningModule):
    def __init__(self, num_players=100, num_teams=2, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.patch_encoder = PatchEncoder()
        self.heads = MultiTaskHeads(emb_dim=4096)
        # 自己教師あり: 軌道予測のMSE
        self.criterion_delta = nn.MSELoss(reduction='none')
        # コントラスト学習
        self.tau = 0.07
        self.lr = lr
        # ロス重み
        self.lambda_delta = 1.0
        self.lambda_infonce = 0.5

    def forward(self, patch):
        emb = F.normalize(self.patch_encoder(patch), dim=-1)
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
        emb = F.normalize(self.patch_encoder(patch), dim=-1)
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
        emb = F.normalize(self.patch_encoder(patch), dim=-1)
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
        emb = F.normalize(self.patch_encoder(patch), dim=-1)
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