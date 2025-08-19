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
   - Default LM: Japanese `rinna/japanese-gpt2-medium` (no translation bridge required)

## Quick Start (Inference)

```
bash scripts/infer_style.sh
```

Defaults are configurable at the top of `scripts/infer_style.sh`:

- `DEFAULT_MODEL` (default: `rinna/japanese-gpt2-medium`)
- `DEFAULT_MAX_NEW_TOKENS` (default: 128)
- `DEFAULT_PLAYER_ID` (set to a known id to skip name resolution)
- Decoding: `DEFAULT_TEMPERATURE`, `DEFAULT_TOP_P`, `DEFAULT_REPETITION_PENALTY`, `DEFAULT_NO_REPEAT_NGRAM_SIZE`
- Prompt template toggle: `DEFAULT_USE_PROMPT_TEMPLATE` (on by default)

Or call Python directly with full control:

```
python -m Player2Vec.infer_style \
  --question "川崎フロンターレの家長 昭博選手について教えて" \
  --emb checkpoints/player_embeddings.pt \
  --model rinna/japanese-gpt2-medium \
  --player_id 4609 \
  --max_new_tokens 128 \
  --use_prompt_template \
  --temperature 0.6 --top_p 0.9 \
  --repetition_penalty 1.3 --no_repeat_ngram_size 4
```

## Notes

- The style LM receives the player embedding `p` (dim=256, L2-normalized) projected to the LM hidden size and prepended as a soft prefix to the question tokens.
- For non-Japanese LMs, you may enable the JA↔EN translation bridge via `--bridge_ja_en` and specify MT models `--mt_ja_en/--mt_en_ja` (kept optional).
