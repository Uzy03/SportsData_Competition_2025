#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/build_embeddings.sh <CKPT_PATH> [FEATHER_DIR] [OUT_DIR]
# Example:
#   scripts/build_embeddings.sh checkpoints/last.ckpt Preprocessed_data/feather checkpoints

CKPT_PATH=${1:-checkpoints/soccer_model_epoch=11_val_loss=0.6879.ckpt}
FEATHER_DIR=${2:-Preprocessed_data/feather}
OUT_DIR=${3:-checkpoints}

# Ensure Python can import the project package (works in Docker/workspace)
export PYTHONPATH=${PYTHONPATH:-$(pwd)}

mkdir -p "$OUT_DIR"

python -m Player2Vec.build_player_embeddings \
  --feather_dir "$FEATHER_DIR" \
  --ckpt "$CKPT_PATH" \
  --win 150 \
  --stride 75 \
  --out_pt "$OUT_DIR/player_embeddings.pt" \
  --out_csv "$OUT_DIR/player_embeddings.csv"

echo "Embeddings written to: $OUT_DIR/player_embeddings.pt and .csv"
