#!/usr/bin/env bash
set -euo pipefail

# End-to-end pipeline:
# 1) Train representation model (Player2Vec)
# 2) Build player embeddings from trained checkpoint
# 3) Train p_proj for instruction-following LM
# 4) (Optional) Run inference with/without prefix via scripts/infer_style.sh
#
# Usage:
#   scripts/run_full_pipeline.sh \
#     --data <TRAIN_DATA_DIR> \
#     --feather <PREPROCESSED_FEATHER_DIR> \
#     --epochs 10 \
#     --batch 64 \
#     --pproj_csv <CSV_WITH_id_prompt_target> \
#     --lm "Qwen/Qwen2.5-1.5B-Instruct" \
#     --out checkpoints
#
# Notes:
# - Requires Python deps installed and GPU if available.
# - Produces:
#   - Player2Vec checkpoints: Player2Vec/checkpoints/
#   - Player embeddings: <OUT>/player_embeddings.pt/.csv
#   - p_proj checkpoint: <OUT>/p_proj.pt

DATA_DIR=""
FEATHER_DIR="Preprocessed_data/feather"
EPOCHS=10
BATCH=64
PPROJ_CSV=""
LM_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
OUT_DIR="checkpoints"
PREFIX_SCALE=0.05
PREFIX_LEN=8

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) DATA_DIR="$2"; shift 2;;
    --feather) FEATHER_DIR="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --batch) BATCH="$2"; shift 2;;
    --pproj_csv) PPROJ_CSV="$2"; shift 2;;
    --lm) LM_MODEL="$2"; shift 2;;
    --out) OUT_DIR="$2"; shift 2;;
    --prefix_scale) PREFIX_SCALE="$2"; shift 2;;
    --prefix_len) PREFIX_LEN="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "${DATA_DIR}" ]]; then
  echo "[ERROR] --data <TRAIN_DATA_DIR> is required" >&2
  exit 1
fi
if [[ -z "${PPROJ_CSV}" ]]; then
  echo "[WARN] --pproj_csv not provided; p_proj training will be skipped"
fi

# Ensure imports work when running from repo root
export PYTHONPATH=${PYTHONPATH:-$(pwd)}

mkdir -p "${OUT_DIR}"

# 1) Train representation (saves to Player2Vec/checkpoints)
python -m Player2Vec.trainer.train fit \
  --data "${DATA_DIR}" \
  --epochs "${EPOCHS}" \
  --batch "${BATCH}"

# Resolve last.ckpt path produced by Lightning
REP_CKPT="Player2Vec/checkpoints/last.ckpt"
if [[ ! -f "${REP_CKPT}" ]]; then
  echo "[WARN] ${REP_CKPT} not found, attempting to locate a recent checkpoint..." >&2
  REP_CKPT=$(ls -t Player2Vec/checkpoints/*.ckpt 2>/dev/null | head -n1 || true)
fi
if [[ -z "${REP_CKPT}" || ! -f "${REP_CKPT}" ]]; then
  echo "[ERROR] No representation checkpoint found under Player2Vec/checkpoints" >&2
  exit 1
fi

echo "[INFO] Using representation ckpt: ${REP_CKPT}" >&2

# 2) Build embeddings
bash scripts/build_embeddings.sh "${REP_CKPT}" "${FEATHER_DIR}" "${OUT_DIR}"

# 3) Train p_proj (optional)
if [[ -n "${PPROJ_CSV}" ]]; then
  python -m Player2Vec.trainer.p_proj_trainer \
    --data_csv "${PPROJ_CSV}" \
    --emb "${OUT_DIR}/player_embeddings.pt" \
    --model "${LM_MODEL}" \
    --epochs 3 \
    --batch 2 \
    --prefix_scale "${PREFIX_SCALE}" \
    --prefix_len "${PREFIX_LEN}" \
    --save_ckpt "${OUT_DIR}/p_proj.pt"
else
  echo "[INFO] Skipping p_proj training (no --pproj_csv)"
fi

echo "[DONE] Pipeline complete. Outputs in ${OUT_DIR}/"
