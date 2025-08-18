#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/infer_style.sh "<QUESTION>" [EMB_PATH] [PLAYERS_CSV] [MODEL] [MAX_NEW_TOKENS] [PLAYER_ID]
# Example:
#   scripts/infer_style.sh "東京FCの田中選手について教えて" checkpoints/player_embeddings.pt data/players.csv distilgpt2 64
#   # or specify player id directly to skip name resolution
#   scripts/infer_style.sh "田中選手について教えて" checkpoints/player_embeddings.pt _ distilgpt2 64 24
# Notes:
#   - You can hardcode defaults below (DEFAULT_*). CLI args override them.
#   - Ensure you run this from the project root so PYTHONPATH is set correctly.

# ---- Hardcoded defaults (edit here) ----
DEFAULT_QUESTION="川崎フロンターレの家長　昭博選手について教えて"
DEFAULT_EMB_PATH="checkpoints/player_embeddings.pt"
DEFAULT_PLAYERS_CSV="data/players.csv"
DEFAULT_MODEL="distilgpt2"
DEFAULT_MAX_NEW_TOKENS="64"
# If set (non-empty), skip name resolution and use this id directly
DEFAULT_PLAYER_ID="4609"
# ----------------------------------------

QUESTION=${1:-${DEFAULT_QUESTION}}
EMB_PATH=${2:-${DEFAULT_EMB_PATH}}
PLAYERS_CSV=${3:-${DEFAULT_PLAYERS_CSV}}
MODEL=${4:-${DEFAULT_MODEL}}
MAX_NEW_TOKENS=${5:-${DEFAULT_MAX_NEW_TOKENS}}
PLAYER_ID=${6:-${DEFAULT_PLAYER_ID}}

if [[ -z "${QUESTION}" ]]; then
  echo "ERROR: QUESTION is required (either pass as arg or set DEFAULT_QUESTION)."
  echo "Usage: scripts/infer_style.sh \"<QUESTION>\" [EMB_PATH] [PLAYERS_CSV] [MODEL] [MAX_NEW_TOKENS] [PLAYER_ID]"
  exit 1
fi

# Ensure Python can import the project package (works in Docker/workspace)
export PYTHONPATH=${PYTHONPATH:-$(pwd)}

# Build command
CMD=(python -m Player2Vec.infer_style \
  --question "${QUESTION}" \
  --emb "${EMB_PATH}" \
  --model "${MODEL}" \
  --max_new_tokens "${MAX_NEW_TOKENS}")

if [[ -n "${PLAYER_ID}" ]]; then
  # Use explicit player id; ignore players_csv
  CMD+=(--player_id "${PLAYER_ID}")
else
  # Use name resolution via players_csv unless placeholder '_' is given
  if [[ "${PLAYERS_CSV}" != "_" && -n "${PLAYERS_CSV}" ]]; then
    CMD+=(--players_csv "${PLAYERS_CSV}")
  fi
fi

"${CMD[@]}"
