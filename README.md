## Build & Run (Docker)

```
docker build -t soccer-player2vec .
docker run -it --name soccer-player2vec --gpus all -v "$PWD":/workspace soccer-player2vec /bin/bash
```

## Requirements (local/dev)

Key Python deps are pinned in `requirements.txt`. For Japanese models/tokenizers, `sentencepiece` is required.

```
pip install -r requirements.txt
```

## End-to-End Pipeline

1. Data preprocessing
   - Source: `Dataprocesser/`, `DataPreProcessing/`
   - Goal: produce training-ready data for Player2Vec.

2. Train Player2Vec (embeddings)
   - Code: `Player2Vec/datamodules/datamodule.py`, `Player2Vec/trainer/train.py`, `Player2Vec/models/model.py`
   - Output: `checkpoints/player_embeddings.pt` (dict[player_id] -> Tensor(256))
   - Helper script: `scripts/build_embeddings.sh` (adjust paths as needed)

3. Inference (natural language style description)
   - Script: `scripts/infer_style.sh`
   - Implementation: `Player2Vec/infer_style.py` + `Player2Vec/models/model.py`
   - Default LM: English `distilgpt2` with JA↔EN bridge enabled by default

## Quick Start (Inference)

```
bash scripts/infer_style.sh
```

Defaults are configurable at the top of `scripts/infer_style.sh`:

- `DEFAULT_MODEL` (default: `distilgpt2`)
- `DEFAULT_MAX_NEW_TOKENS` (default: 128)
- `DEFAULT_PLAYER_ID` (set to a known id to skip name resolution)
- Decoding: `DEFAULT_TEMPERATURE`, `DEFAULT_TOP_P`, `DEFAULT_REPETITION_PENALTY`, `DEFAULT_NO_REPEAT_NGRAM_SIZE`
- Prompt template toggle: `DEFAULT_USE_PROMPT_TEMPLATE` (on by default)
- JA↔EN bridge: `DEFAULT_BRIDGE_JA_EN` (on by default)

Or call Python directly with full control:

```
python -m Player2Vec.infer_style \
  --question "川崎フロンターレの家長 昭博選手について教えて" \
  --emb checkpoints/player_embeddings.pt \
  --model distilgpt2 \
  --player_id 4609 \
  --max_new_tokens 128 \
  --use_prompt_template \
  --temperature 0.7 --top_p 0.9 \
  --repetition_penalty 1.2 --no_repeat_ngram_size 3 \
  --bridge_ja_en \
  --mt_ja_en staka/fugumt-ja-en \
  --mt_en_ja staka/fugumt-en-ja
```

## Notes

- The style LM receives the player embedding `p` (dim=256, L2-normalized) projected to the LM hidden size and prepended as a soft prefix to the question tokens.
- By default we generate with an English LM and translate JA→EN for input and EN→JA for output. You can disable the bridge by omitting `--bridge_ja_en`.

## Model Architecture (overview)

- __PatchEncoder__ (`Player2Vec/models/model.py`):
  - CNN (1D conv) extracts per-time-step spatial features from tracking `(B, 23, T, 2)`.
  - Bi-LSTM summarizes temporal dynamics per player; mean pooling across 23 players.
  - Linear head → 4096-dim representation, then L2 normalize to obtain player embedding `p` (internally reduced to 256-dim for style LM).

- __MultiTaskHeads__ (`Player2Vec/models/model.py`):
  - Predicts last-steps deltas as a self-supervised task to shape the representation.

- __SLMWrapper__ (`Player2Vec/models/model.py`):
  - Loads a causal LM (default `distilgpt2`).
  - Projects `p` via `p_proj(256→hidden)` and injects as a soft prefix to token embeddings.
  - Decoding controlled by `temperature`, `top_p`, `repetition_penalty`, `no_repeat_ngram_size`.
  - Optional JA↔EN machine translation in `Player2Vec/infer_style.py` using seq2seq MT models.
