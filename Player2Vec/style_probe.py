import argparse
import os
from typing import Dict, Optional, Tuple, List

import torch
import numpy as np

from Player2Vec.models.model import SLMWrapper


def load_embeddings(emb_path: str) -> Dict[str, torch.Tensor]:
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embeddings not found: {emb_path}")
    ext = os.path.splitext(emb_path)[1].lower()
    if ext in [".pt", ".pth"]:
        obj = torch.load(emb_path, map_location="cpu")
        if not isinstance(obj, dict):
            raise ValueError("Expected dict id->tensor in .pt/.pth")
        return {str(k): torch.as_tensor(v).float().view(-1) for k, v in obj.items()}
    elif ext == ".csv":
        import csv
        out: Dict[str, torch.Tensor] = {}
        with open(emb_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = str(row.get("id"))
                vec = [float(v) for k, v in row.items() if k != "id" and v not in (None, "")]
                out[pid] = torch.tensor(vec, dtype=torch.float32)
        return out
    else:
        raise ValueError(f"Unsupported format: {ext}")


def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=dim)


def default_prompt() -> str:
    return (
        "[INST] あなたはサッカーアナリストです。\n"
        "以下の情報を読み取り、直後の <STYLE> の内部表現に\n"
        "選手のプレースタイルを要約して保持してください。\n"
        "<STYLE> その後のテキストは無視して構いません。 [/INST]"
    )


def compute_style_vectors(
    emb_path: str,
    out_dir: str = "checkpoints",
    model_name: str = "distilgpt2",
    prefix_scale: float = 0.05,
    prefix_len: int = 8,
    no_prefix: bool = False,
    prompt: Optional[str] = None,
    seed: Optional[int] = 42,
) -> Dict[str, torch.Tensor]:
    os.makedirs(out_dir, exist_ok=True)
    embeds = load_embeddings(emb_path)

    # Prepare wrapper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    slm = SLMWrapper(model_name=model_name).to(device)

    if prompt is None:
        prompt = default_prompt()

    style_map: Dict[str, torch.Tensor] = {}
    for pid, p in embeds.items():
        # ensure 256-dim
        p = torch.as_tensor(p, dtype=torch.float32)
        if p.numel() < 256:
            p = torch.cat([p, torch.zeros(256 - p.numel())], dim=0)
        elif p.numel() > 256:
            p = p[:256]
        p = l2_normalize(p, dim=0)
        vec = slm.style_probe(
            p.to(device),
            prompt,
            style_token="<STYLE>",
            prefix_scale=prefix_scale,
            prefix_len=prefix_len,
            no_prefix=no_prefix,
            seed=seed,
        ).detach().cpu()
        # final l2 normalize
        vec = l2_normalize(vec, dim=0)
        style_map[str(pid)] = vec

    # Save .pt
    pt_path = os.path.join(out_dir, "style_vectors.pt")
    torch.save(style_map, pt_path)

    # Save .csv
    csv_path = os.path.join(out_dir, "style_vectors.csv")
    import csv
    hidden = next(iter(style_map.values())).numel()
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["id"] + [f"s{i}" for i in range(hidden)]
        writer.writerow(header)
        for pid, v in style_map.items():
            writer.writerow([pid] + [float(x) for x in v.tolist()])

    return style_map


def visualize(
    style_map: Dict[str, torch.Tensor],
    out_dir: str,
    method: str = "umap",
    metric: str = "cosine",
    seed: int = 42,
):
    os.makedirs(out_dir, exist_ok=True)
    X = torch.stack([v for _, v in style_map.items()], dim=0).numpy()
    ids = list(style_map.keys())

    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric=metric, random_state=seed)
            Z = reducer.fit_transform(X)
        except Exception:
            # fallback to TSNE if umap not available
            from sklearn.manifold import TSNE
            Z = TSNE(n_components=2, random_state=seed, metric="cosine").fit_transform(X)
    else:
        from sklearn.manifold import TSNE
        Z = TSNE(n_components=2, random_state=seed, metric="cosine").fit_transform(X)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.scatter(Z[:, 0], Z[:, 1], s=12, alpha=0.8)
    for i, pid in enumerate(ids):
        if i % max(1, len(ids)//50) == 0:  # thin labels
            plt.text(Z[i, 0], Z[i, 1], str(pid), fontsize=6, alpha=0.7)
    out_path = os.path.join(out_dir, f"style_{method}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return Z, ids


def _load_labels(labels_csv: str, id_col: str = "id", label_col: str = "label") -> Dict[str, str]:
    import csv
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"labels_csv not found: {labels_csv}")
    m: Dict[str, str] = {}
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = str(row[id_col])
            lab = str(row[label_col])
            m[pid] = lab
    return m


def _neighbor_purity(X: np.ndarray, labels: List[str], k: int = 10, metric: str = "cosine") -> float:
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    lab = np.array(labels)
    nn = NearestNeighbors(n_neighbors=min(k+1, len(X)), metric=metric)
    nn.fit(X)
    dists, idx = nn.kneighbors(X, return_distance=True)
    # skip self at index 0
    idx = idx[:, 1:]
    same = 0
    total = 0
    for i in range(len(X)):
        neigh = idx[i]
        same += (lab[neigh] == lab[i]).sum()
        total += len(neigh)
    return float(same) / float(max(total, 1))


def compute_metrics(
    vecs: Dict[str, torch.Tensor],
    labels_map: Dict[str, str],
    metric_space: str = "cosine",
    knn_k: int = 10,
) -> Dict[str, float]:
    from sklearn.metrics import silhouette_score, normalized_mutual_info_score
    from sklearn.cluster import KMeans
    import numpy as np

    ids = [pid for pid in vecs.keys() if pid in labels_map]
    if not ids:
        raise ValueError("No overlapping ids between embeddings and labels_csv")
    X = torch.stack([vecs[pid] for pid in ids], dim=0).numpy()
    y = [labels_map[pid] for pid in ids]

    # Neighbor purity on high-dim vectors
    purity = _neighbor_purity(X, y, k=knn_k, metric=metric_space)

    # Silhouette (requires numeric labels)
    # Map labels to integers
    uniq = {lab: i for i, lab in enumerate(sorted(set(y)))}
    y_int = np.array([uniq[l] for l in y])
    try:
        sil = float(silhouette_score(X, y_int, metric=metric_space))
    except Exception:
        sil = float("nan")

    # NMI via KMeans with n_clusters = n_labels
    kmeans = KMeans(n_clusters=len(uniq), n_init=10, random_state=42)
    pred = kmeans.fit_predict(X)
    nmi = float(normalized_mutual_info_score(y_int, pred))

    return {"neighbor_purity": purity, "silhouette": sil, "nmi": nmi}


def main():
    parser = argparse.ArgumentParser(description="StyleProbe: extract style embeddings using soft prefix and fixed prompt")
    parser.add_argument("--emb", type=str, default="checkpoints/player_embeddings.pt", help="Path to p embeddings (.pt/.pth/.csv)")
    parser.add_argument("--out", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--model", type=str, default="distilgpt2", help="HF Causal LM model name")
    parser.add_argument("--prefix_scale", type=float, default=0.05)
    parser.add_argument("--prefix_len", type=int, default=8)
    parser.add_argument("--no_prefix", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=str, default=None, help="Override fixed Japanese prompt")
    parser.add_argument("--plot", action="store_true", help="Run UMAP/t-SNE and save a 2D scatter image")
    parser.add_argument("--method", type=str, default="umap", choices=["umap", "tsne"]) 
    parser.add_argument("--metric", type=str, default="cosine")
    parser.add_argument("--plot_baseline_p", action="store_true", help="Also plot UMAP/t-SNE of raw p embeddings for comparison")
    parser.add_argument("--metrics", action="store_true", help="Compute cluster quality metrics if labels are provided")
    parser.add_argument("--labels_csv", type=str, default=None, help="CSV with columns id,label for cluster metrics")
    parser.add_argument("--label_col", type=str, default="label", help="Label column name in labels_csv")
    parser.add_argument("--knn_k", type=int, default=10, help="k for neighbor purity metric")
    args = parser.parse_args()

    style_map = compute_style_vectors(
        emb_path=args.emb,
        out_dir=args.out,
        model_name=args.model,
        prefix_scale=args.prefix_scale,
        prefix_len=args.prefix_len,
        no_prefix=args.no_prefix,
        prompt=args.prompt,
        seed=args.seed,
    )

    if args.plot:
        Zs, ids_s = visualize(style_map, out_dir=args.out, method=args.method, metric=args.metric, seed=args.seed)
        print(f"Saved 2D plot to {os.path.join(args.out, f'style_{args.method}.png')}")

    # Baseline: plot p embeddings
    if args.plot_baseline_p:
        # load raw p
        embeds = load_embeddings(args.emb)
        # normalize and align dims
        p_map: Dict[str, torch.Tensor] = {}
        for pid, p in embeds.items():
            p = torch.as_tensor(p, dtype=torch.float32)
            if p.numel() < 256:
                p = torch.cat([p, torch.zeros(256 - p.numel())], dim=0)
            elif p.numel() > 256:
                p = p[:256]
            p_map[str(pid)] = l2_normalize(p, dim=0)
        Zp, ids_p = visualize(p_map, out_dir=args.out, method=args.method, metric=args.metric, seed=args.seed)
        # rename output file to baseline name
        base_png = os.path.join(args.out, f"style_{args.method}.png")
        baseline_png = os.path.join(args.out, f"baseline_p_{args.method}.png")
        if os.path.exists(base_png):
            try:
                os.replace(base_png, os.path.join(args.out, f"style_{args.method}.png"))
            except Exception:
                pass
        # save baseline separately by re-plotting
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.scatter(Zp[:, 0], Zp[:, 1], s=12, alpha=0.8)
        for i, pid in enumerate(ids_p):
            if i % max(1, len(ids_p)//50) == 0:
                plt.text(Zp[i, 0], Zp[i, 1], str(pid), fontsize=6, alpha=0.7)
        plt.tight_layout()
        plt.savefig(baseline_png, dpi=200)
        plt.close()
        print(f"Saved baseline 2D plot to {baseline_png}")

    # Metrics (requires labels)
    if args.metrics and args.labels_csv:
        labels_map = _load_labels(args.labels_csv, id_col="id", label_col=args.label_col)
        # metrics on style vectors
        m_style = compute_metrics(style_map, labels_map, metric_space=args.metric, knn_k=args.knn_k)
        # metrics on baseline p
        embeds = load_embeddings(args.emb)
        p_map: Dict[str, torch.Tensor] = {}
        for pid, p in embeds.items():
            p = torch.as_tensor(p, dtype=torch.float32)
            if p.numel() < 256:
                p = torch.cat([p, torch.zeros(256 - p.numel())], dim=0)
            elif p.numel() > 256:
                p = p[:256]
            p_map[str(pid)] = l2_normalize(p, dim=0)
        m_base = compute_metrics(p_map, labels_map, metric_space=args.metric, knn_k=args.knn_k)
        import json
        out_json = os.path.join(args.out, "style_metrics.json")
        with open(out_json, "w") as f:
            json.dump({"style": m_style, "baseline_p": m_base}, f, ensure_ascii=False, indent=2)
        print(f"Saved metrics to {out_json}")
    elif args.metrics and not args.labels_csv:
        print("[WARN] --metrics specified but --labels_csv missing; skipping metrics computation")

    print(f"Saved style vectors to {os.path.join(args.out, 'style_vectors.pt')} and .csv")


if __name__ == "__main__":
    main()
