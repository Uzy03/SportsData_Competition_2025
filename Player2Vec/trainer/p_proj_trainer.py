import argparse
import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import csv

from Player2Vec.models.model import SLMWrapper


def load_embeddings(emb_path: str) -> Dict[str, torch.Tensor]:
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embeddings not found: {emb_path}")
    ext = os.path.splitext(emb_path)[1].lower()
    if ext in [".pt", ".pth"]:
        obj = torch.load(emb_path, map_location="cpu")
        if not isinstance(obj, dict):
            raise ValueError("Expected dict id->tensor in embeddings .pt/.pth")
        out: Dict[str, torch.Tensor] = {}
        for k, v in obj.items():
            t = torch.as_tensor(v).float().view(-1)
            out[str(k)] = t
        return out
    if ext == ".csv":
        out: Dict[str, torch.Tensor] = {}
        with open(emb_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = str(row.get("id"))
                vec = []
                for k, v in row.items():
                    if k == "id":
                        continue
                    if v is None or v == "":
                        continue
                    vec.append(float(v))
                out[pid] = torch.tensor(vec, dtype=torch.float32)
        return out
    raise ValueError(f"Unsupported embedding format: {ext}")


def select_p(embeds: Dict[str, torch.Tensor], player_id: str) -> torch.Tensor:
    if player_id in embeds:
        return embeds[player_id]
    if player_id.isdigit() and str(int(player_id)) in embeds:
        return embeds[str(int(player_id))]
    if player_id.lstrip("0") in embeds:
        return embeds[player_id.lstrip("0")]
    raise KeyError(f"player_id '{player_id}' not found in embeddings")


class PProjDataset(Dataset):
    """
    Expects a CSV file with columns: id,prompt,target
    id: player id key to look up p
    prompt: prompt text (conditioning)
    target: target text to predict (teacher forcing)
    """
    def __init__(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)
        self.rows: List[Dict[str, str]] = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = str(row.get("id"))
                pr = (row.get("prompt") or "").strip()
                tg = (row.get("target") or "").strip()
                if not pid or not pr or not tg:
                    # skip incomplete rows
                    continue
                self.rows.append({"id": pid, "prompt": pr, "target": tg})
        if not self.rows:
            raise ValueError("No valid rows found in CSV (need columns id,prompt,target)")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.rows[idx]


class PProjLightningModule(pl.LightningModule):
    def __init__(self, model_name: str, emb_path: str, lr: float = 1e-3, prefix_scale: float = 0.05, prefix_len: int = 8):
        super().__init__()
        self.save_hyperparameters()
        self.slm = SLMWrapper(model_name=model_name)
        # freeze LM
        for p in self.slm.model.parameters():
            p.requires_grad = False
        # train only p_proj and its LN
        for p in self.slm.p_proj.parameters():
            p.requires_grad = True
        for p in self.slm.prefix_ln.parameters():
            p.requires_grad = True
        self.embeds = load_embeddings(emb_path)
        self.lr = lr
        self.prefix_scale = prefix_scale
        self.prefix_len = prefix_len

    def configure_optimizers(self):
        params = list(self.slm.p_proj.parameters()) + list(self.slm.prefix_ln.parameters())
        return torch.optim.Adam(params, lr=self.lr)

    def training_step(self, batch, batch_idx):
        device = self.device
        tokenizer = self.slm.tokenizer
        model = self.slm.model
        batch_size = len(batch["id"]) if isinstance(batch["id"], list) else len(batch["id"])

        # Build per-sample inputs
        input_embeds_list = []
        labels_list = []
        for i in range(batch_size):
            pid = batch["id"][i]
            prompt = batch["prompt"][i]
            target = batch["target"][i]

            # p (256-dim, l2 normalize)
            p = select_p(self.embeds, str(pid))
            p = p[:256] if p.numel() > 256 else torch.nn.functional.pad(p, (0, max(0, 256 - p.numel())), value=0.0)
            p = torch.nn.functional.normalize(p, dim=0).to(device)
            p = p.unsqueeze(0)  # (1,256)

            # tokenize prompt & target
            pr = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            tg = tokenizer(target, return_tensors="pt", add_special_tokens=False)
            pr_ids = pr.input_ids.to(device)
            tg_ids = tg.input_ids.to(device)

            # build full ids (prompt + target)
            all_ids = torch.cat([pr_ids, tg_ids], dim=1)  # (1, L)
            all_embeds = model.get_input_embeddings()(all_ids)

            # soft prefix
            prefix_vec = self.slm.prefix_ln(self.slm.p_proj(p)) * self.prefix_scale
            soft_prefix = prefix_vec.unsqueeze(1).expand(-1, int(max(self.prefix_len, 1)), -1)
            all_embeds = torch.cat([soft_prefix, all_embeds], dim=1)  # (1, P+L, H)

            # labels: -100 for prefix and prompt, target ids for target positions
            labels = torch.full((1, all_embeds.size(1)), -100, dtype=torch.long, device=device)
            start_tgt = soft_prefix.size(1) + pr_ids.size(1)
            labels[:, start_tgt:start_tgt + tg_ids.size(1)] = tg_ids

            input_embeds_list.append(all_embeds)
            labels_list.append(labels)

        # pad to max length in batch
        max_len = max(x.size(1) for x in input_embeds_list)
        H = input_embeds_list[0].size(-1)
        padded_embeds = []
        padded_labels = []
        attention_masks = []
        for emb, lab in zip(input_embeds_list, labels_list):
            pad_len = max_len - emb.size(1)
            if pad_len > 0:
                pad_emb = torch.zeros((emb.size(0), pad_len, H), device=device, dtype=emb.dtype)
                emb = torch.cat([emb, pad_emb], dim=1)
                lab = torch.cat([lab, torch.full((lab.size(0), pad_len), -100, dtype=lab.dtype, device=device)], dim=1)
            padded_embeds.append(emb)
            padded_labels.append(lab)
            attention_masks.append(torch.ones((emb.size(0), emb.size(1)), dtype=torch.long, device=device))

        inputs_embeds = torch.cat(padded_embeds, dim=0)
        labels = torch.cat(padded_labels, dim=0)
        attention_mask = torch.cat(attention_masks, dim=0)

        out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        loss = out.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss


def collate_fn(samples: List[Dict[str, str]]) -> Dict[str, List[str]]:
    out = {"id": [], "prompt": [], "target": []}
    for s in samples:
        out["id"].append(s["id"])
        out["prompt"].append(s["prompt"])
        out["target"].append(s["target"])
    return out


def main():
    parser = argparse.ArgumentParser(description="Train p_proj for SLMWrapper with teacher-forced targets")
    parser.add_argument("--data_csv", type=str, required=True, help="CSV with columns id,prompt,target")
    parser.add_argument("--emb", type=str, required=True, help="Path to player embeddings (.pt/.pth/.csv)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--prefix_scale", type=float, default=0.05)
    parser.add_argument("--prefix_len", type=int, default=8)
    parser.add_argument("--save_ckpt", type=str, default="checkpoints/p_proj.pt")
    args = parser.parse_args()

    dm = PProjDataset(args.data_csv)
    dl = DataLoader(dm, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)

    module = PProjLightningModule(
        model_name=args.model,
        emb_path=args.emb,
        lr=args.lr,
        prefix_scale=args.prefix_scale,
        prefix_len=args.prefix_len,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = module.to(device)

    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="auto", log_every_n_steps=1)
    trainer.fit(module, train_dataloaders=dl)

    # Save only p_proj (and LN) weights
    os.makedirs(os.path.dirname(args.save_ckpt), exist_ok=True)
    state = {
        "p_proj.weight": module.slm.p_proj.weight.detach().cpu(),
        "p_proj.bias": module.slm.p_proj.bias.detach().cpu(),
        "prefix_ln.weight": module.slm.prefix_ln.weight.detach().cpu(),
        "prefix_ln.bias": module.slm.prefix_ln.bias.detach().cpu(),
    }
    torch.save(state, args.save_ckpt)
    print(f"Saved p_proj checkpoint to {args.save_ckpt}")


if __name__ == "__main__":
    main()
