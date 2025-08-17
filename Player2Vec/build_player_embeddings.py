import argparse
import os
import glob
import ast
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from Player2Vec.models.model import SoccerLightningModule


def load_attack_from_feather(feather_dir: str, stem: str) -> Tuple[np.ndarray, List, np.ndarray]:
    """
    Reconstruct (23, T, 2) tensor, entities list, frames array from feather files.
    - feather_dir: directory containing subdirs 'tensor' and 'meta'
    - stem: e.g., 'attack_2022091601_0001'
    Returns: (tensor, entities, frames)
    """
    tensor_path = os.path.join(feather_dir, 'tensor', f'{stem}.feather')
    meta_path = os.path.join(feather_dir, 'meta', f'{stem}.feather')

    tensor_df = pd.read_feather(tensor_path)
    meta_df = pd.read_feather(meta_path)

    shape = (
        int(meta_df['tensor_shape_0'].iloc[0]),
        int(meta_df['tensor_shape_1'].iloc[0]),
        int(meta_df['tensor_shape_2'].iloc[0]),
    )
    tensor = np.full(shape, np.nan, dtype=np.float32)
    for _, row in tensor_df.iterrows():
        ei = int(row['entity_idx'])
        fi = int(row['frame_idx'])
        tensor[ei, fi, 0] = float(row['x'])
        tensor[ei, fi, 1] = float(row['y'])

    entities = ast.literal_eval(meta_df['entities'].iloc[0])
    frames = np.array(ast.literal_eval(meta_df['frames'].iloc[0]))
    return tensor, entities, frames


essential_pad_value = 0.0

def make_windows(x: np.ndarray, win: int = 150, stride: int = 75) -> List[Tuple[int, int]]:
    T = x.shape[1]
    if T <= 0:
        return []
    idx = []
    start = 0
    while start < T:
        end = min(start + win, T)
        idx.append((start, end))
        if end == T:
            break
        start += stride
    return idx


def compute_player_window_embeddings(patch_encoder, window: torch.Tensor) -> torch.Tensor:
    """
    Compute per-player 4096-d embeddings for a single window (1,23,T,2) by
    mirroring PatchEncoder internals, applying fc to each player's final hidden.
    Returns: (23, 4096) tensor (players with all-NaN/zeros may become zeros).
    """
    patch_encoder.eval()
    device = next(patch_encoder.parameters()).device
    B, P, T, C = window.shape
    assert B == 1 and C == 2

    outs = []
    for i in range(P):
        player_data = window[:, i, :, :]  # (1, T, 2)
        # replace NaN with 0
        player_data = torch.nan_to_num(player_data, nan=0.0)
        spatial_feat = patch_encoder.spatial_conv(player_data.transpose(1, 2))  # (1, hidden, T)
        spatial_feat = spatial_feat.transpose(1, 2)  # (1, T, hidden)
        _, (hidden, _) = patch_encoder.lstm(spatial_feat)
        if patch_encoder.lstm.bidirectional:
            final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            final_hidden = hidden[-1]
        emb4096 = patch_encoder.fc(patch_encoder.dropout(final_hidden))  # (1,4096)
        outs.append(emb4096.squeeze(0))
    return torch.stack(outs, dim=0)  # (23,4096)


def pca_project(X: torch.Tensor, k: int = 256) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    X: (N, D). Returns projected (N, k) and params {mean, components}.
    PCA via SVD on centered X.
    """
    Xc = X - X.mean(dim=0, keepdim=True)
    # economy SVD via torch.linalg.svd; for large D this may be heavy but OK here
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    components = Vh[:k, :]  # (k, D)
    Xk = Xc @ components.T  # (N, k)
    params = {"mean": X.mean(dim=0), "components": components}
    return Xk, params


def main():
    parser = argparse.ArgumentParser(description="Build per-player embeddings p from preprocessed feather and a trained encoder")
    parser.add_argument("--feather_dir", type=str, default="Preprocessed_data/feather", help="Base dir with meta/ and tensor/")
    parser.add_argument("--ckpt", type=str, required=True, help="Lightning checkpoint .ckpt for SoccerLightningModule")
    parser.add_argument("--win", type=int, default=150, help="window length in frames (30s@5fps)")
    parser.add_argument("--stride", type=int, default=75, help="stride in frames (15s@5fps)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out_pt", type=str, default="checkpoints/player_embeddings.pt")
    parser.add_argument("--out_csv", type=str, default="checkpoints/player_embeddings.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_pt), exist_ok=True)

    # collect stems
    meta_glob = os.path.join(args.feather_dir, 'meta', 'attack_*.feather')
    stems = [os.path.splitext(os.path.basename(p))[0] for p in glob.glob(meta_glob)]
    stems.sort()

    # load model and pick device
    model = SoccerLightningModule.load_from_checkpoint(args.ckpt, map_location='cpu')
    model.eval()
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    model = model.to(device)

    # accumulators per player_id -> sum(4096) and count
    sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}

    for stem in tqdm(stems, desc="attacks"):
        try:
            tensor_np, entities, frames = load_attack_from_feather(args.feather_dir, stem)
        except Exception as e:
            print(f"[WARN] failed to load {stem}: {e}")
            continue
        if tensor_np.shape[0] != 23:
            print(f"[WARN] skip {stem}: unexpected entity dim {tensor_np.shape}")
            continue
        # windows
        for (s, e) in make_windows(tensor_np, win=args.win, stride=args.stride):
            slice_np = tensor_np[:, s:e, :]
            # zero pad to win
            T = slice_np.shape[1]
            if T < args.win:
                pad = np.zeros((slice_np.shape[0], args.win - T, slice_np.shape[2]), dtype=np.float32)
                slice_np = np.concatenate([slice_np, pad], axis=1)
            # replace NaNs with 0
            slice_np = np.nan_to_num(slice_np, nan=0.0)
            # to tensor
            window = torch.from_numpy(slice_np).unsqueeze(0).to(device)  # (1,23,win,2)
            with torch.no_grad():
                per_player_4096 = compute_player_window_embeddings(model.patch_encoder, window)  # (23,4096)
            per_player_4096 = F.normalize(per_player_4096, dim=-1)

            # accumulate per actual player id
            for i, pid in enumerate(entities):
                if pid is None or pid == 'ball':
                    continue
                key = str(pid)
                vec = per_player_4096[i].detach().cpu()
                if key not in sums:
                    sums[key] = vec.clone()
                    counts[key] = 1
                else:
                    sums[key] += vec
                    counts[key] += 1

    if not sums:
        raise RuntimeError("No player embeddings accumulated. Check data and entities in meta.")

    # average and stack
    ids = sorted(sums.keys(), key=lambda k: (len(k), k))
    X = torch.stack([sums[k] / counts[k] for k in ids], dim=0)  # (N,4096)

    # PCA -> 256 and L2 normalize
    X256, pca_params = pca_project(X, k=256)
    X256 = F.normalize(X256, dim=-1)

    # save .pt as dict: id -> tensor(256)
    emb_dict: Dict[str, torch.Tensor] = {pid: X256[i] for i, pid in enumerate(ids)}
    torch.save(emb_dict, args.out_pt)

    # save .csv
    arr = X256.numpy()
    cols = [f"p{i}" for i in range(arr.shape[1])]
    df = pd.DataFrame(arr, columns=cols)
    df.insert(0, 'id', ids)
    df.to_csv(args.out_csv, index=False)

    # also save PCA params for reproducibility
    pca_file = os.path.splitext(args.out_pt)[0] + "_pca.pt"
    torch.save({"mean": pca_params["mean"], "components": pca_params["components"], "ids": ids}, pca_file)

    print(f"Saved embeddings: {args.out_pt} and {args.out_csv}")


if __name__ == "__main__":
    main()
