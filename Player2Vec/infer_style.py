import argparse
import os
from typing import Dict, Optional

import torch

from Player2Vec.models.model import SLMWrapper


def _load_embeddings(emb_path: str) -> Dict[str, torch.Tensor]:
    """
    Load player embeddings p (dim=256 expected) from:
    - .pt/.pth: a dict mapping player_id (str or int) -> 1D tensor
    - .csv: header with id plus 256 columns p0..p255 (comma-separated)
    Returns a dict[str, Tensor(256)].
    """
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embeddings not found: {emb_path}")

    ext = os.path.splitext(emb_path)[1].lower()
    if ext in [".pt", ".pth"]:
        obj = torch.load(emb_path, map_location="cpu")
        if isinstance(obj, dict):
            out: Dict[str, torch.Tensor] = {}
            for k, v in obj.items():
                key = str(k)
                t = torch.as_tensor(v).float().view(-1)
                out[key] = t
            return out
        else:
            raise ValueError("Expected a dict in the .pt/.pth file mapping id->tensor")

    if ext == ".csv":
        import csv
        out: Dict[str, torch.Tensor] = {}
        with open(emb_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = str(row.get("id"))
                if pid is None:
                    raise ValueError("CSV must contain an 'id' column")
                # Collect all numeric fields except 'id'
                vec = []
                for k, v in row.items():
                    if k == "id":
                        continue
                    if v is None or v == "":
                        continue
                    vec.append(float(v))
                t = torch.tensor(vec, dtype=torch.float32)
                out[pid] = t
        return out

    raise ValueError(f"Unsupported embedding format: {ext}")


def _select_p(embeds: Dict[str, torch.Tensor], player_id: str) -> torch.Tensor:
    if player_id in embeds:
        return embeds[player_id]
    # try integer key fallback
    if player_id.isdigit() and str(int(player_id)) in embeds:
        return embeds[str(int(player_id))]
    # soft match: try zero-padded and raw
    if player_id.lstrip("0") in embeds:
        return embeds[player_id.lstrip("0")]
    raise KeyError(f"player_id '{player_id}' not found in embeddings. Available keys sample: {list(embeds.keys())[:5]}")


def generate_style(player_id: str, question: str, emb_path: str, model_name: str = "distilgpt2", device: Optional[str] = None, max_new_tokens: int = 64) -> str:
    """
    Load p from emb_path and generate natural language with SLMWrapper.
    emb_path: .pt/.pth (dict id->tensor) or .csv
    player_id: key in the dict (string or int convertible)
    """
    embeds = _load_embeddings(emb_path)
    p = _select_p(embeds, player_id)

    # Ensure 256-dim as per design; if longer, take first 256; if shorter, pad zeros
    dim = p.numel()
    if dim < 256:
        p = torch.cat([p, torch.zeros(256 - dim)], dim=0)
    elif dim > 256:
        p = p[:256]

    # l2 normalize p as specified
    p = torch.nn.functional.normalize(p, dim=0)

    slm = SLMWrapper(model_name=model_name)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    slm = slm.to(device)

    with torch.no_grad():
        text = slm.generate(p.to(device), question, max_new_tokens=max_new_tokens)
    return text


def main():
    parser = argparse.ArgumentParser(description="Player style QA inference")
    parser.add_argument("--player_id", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--emb", type=str, default="checkpoints/player_embeddings.pt", help="Path to p embeddings (.pt/.pth/.csv)")
    parser.add_argument("--model", type=str, default="distilgpt2", help="HF Causal LM model name")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    answer = generate_style(
        player_id=args.player_id,
        question=args.question,
        emb_path=args.emb,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
    )
    print(answer)


if __name__ == "__main__":
    main()
