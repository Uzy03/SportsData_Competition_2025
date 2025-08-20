#!/usr/bin/env bash
set -euo pipefail

# StyleProbe runner
# - Extract style vectors s via soft-prefix + fixed prompt
# - Optionally plot UMAP/TSNE and compute cluster metrics
#
# Usage:
#   scripts/style_probe.sh
#   scripts/style_probe.sh --plot --plot_baseline_p
#   scripts/style_probe.sh --metrics --labels_csv data/labels.csv --label_col position
#
# Outputs (default):
#   checkpoints/style_vectors.pt, checkpoints/style_vectors.csv
#   checkpoints/style_umap.png (or style_tsne.png)
#   checkpoints/baseline_p_umap.png (if --plot_baseline_p)
#   checkpoints/style_metrics.json (if --metrics with labels)

# === Defaults (edit as needed) ===
EMB_PATH="checkpoints/player_embeddings.pt"
OUT_DIR="checkpoints"
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
PREFIX_SCALE=0.05
PREFIX_LEN=8
NO_PREFIX=0
SEED=42
PLOT=0
METHOD="umap"         # or tsne
METRIC="cosine"
PLOT_BASELINE_P=0
METRICS=0
LABELS_CSV=""
LABEL_COL="label"
KNN_K=10

# Parse args (override defaults)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --emb) EMB_PATH="$2"; shift 2;;
    --out) OUT_DIR="$2"; shift 2;;
    --model) MODEL_NAME="$2"; shift 2;;
    --prefix_scale) PREFIX_SCALE="$2"; shift 2;;
    --prefix_len) PREFIX_LEN="$2"; shift 2;;
    --no_prefix) NO_PREFIX=1; shift 1;;
    --seed) SEED="$2"; shift 2;;
    --plot) PLOT=1; shift 1;;
    --method) METHOD="$2"; shift 2;;
    --metric) METRIC="$2"; shift 2;;
    --plot_baseline_p) PLOT_BASELINE_P=1; shift 1;;
    --metrics) METRICS=1; shift 1;;
    --labels_csv) LABELS_CSV="$2"; shift 2;;
    --label_col) LABEL_COL="$2"; shift 2;;
    --knn_k) KNN_K="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

export PYTHONPATH=${PYTHONPATH:-$(pwd)}
mkdir -p "$OUT_DIR"

ARGS=(
  --emb "$EMB_PATH"
  --out "$OUT_DIR"
  --model "$MODEL_NAME"
  --prefix_scale "$PREFIX_SCALE"
  --prefix_len "$PREFIX_LEN"
  --seed "$SEED"
  --method "$METHOD"
  --metric "$METRIC"
)

if [[ "$NO_PREFIX" == 1 ]]; then ARGS+=(--no_prefix); fi
if [[ "$PLOT" == 1 ]]; then ARGS+=(--plot); fi
if [[ "$PLOT_BASELINE_P" == 1 ]]; then ARGS+=(--plot_baseline_p); fi
if [[ "$METRICS" == 1 ]]; then
  ARGS+=(--metrics)
  if [[ -n "$LABELS_CSV" ]]; then
    ARGS+=(--labels_csv "$LABELS_CSV" --label_col "$LABEL_COL" --knn_k "$KNN_K")
  else
    echo "[WARN] --metrics set but --labels_csv is empty; metrics will be skipped by script"
  fi
fi

python -m Player2Vec.style_probe "${ARGS[@]}"
