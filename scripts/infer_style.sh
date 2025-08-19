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
DEFAULT_MODEL="rinna/japanese-gpt2-medium"
DEFAULT_MAX_NEW_TOKENS="128"
# If set (non-empty), skip name resolution and use this id directly
DEFAULT_PLAYER_ID="4609"
# Decoding defaults
DEFAULT_TEMPERATURE="0.7"
DEFAULT_TOP_P="0.9"
DEFAULT_REPETITION_PENALTY="1.2"
DEFAULT_NO_REPEAT_NGRAM_SIZE="3"
# Prompt template (on by default)
DEFAULT_USE_PROMPT_TEMPLATE="1"
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
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature "${DEFAULT_TEMPERATURE}" \
  --top_p "${DEFAULT_TOP_P}" \
  --repetition_penalty "${DEFAULT_REPETITION_PENALTY}" \
  --no_repeat_ngram_size "${DEFAULT_NO_REPEAT_NGRAM_SIZE}")

# Toggle prompt template
if [[ "${DEFAULT_USE_PROMPT_TEMPLATE}" == "1" ]]; then
  CMD+=(--use_prompt_template)
fi

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
