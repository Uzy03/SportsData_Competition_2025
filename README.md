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
   - Default LM: `Qwen/Qwen2.5-1.5B-Instruct` (Japanese-capable, instruction following)
   - JA↔EN bridge: disabled by default

## Quick Start (Inference)

```
bash scripts/infer_style.sh
```

Defaults are configurable at the top of `scripts/infer_style.sh`:

- `DEFAULT_MODEL` (default: `Qwen/Qwen2.5-1.5B-Instruct`)
- `DEFAULT_MAX_NEW_TOKENS` (default: 128)
- `DEFAULT_PLAYER_ID` (set to a known id to skip name resolution)
- Decoding: `DEFAULT_TEMPERATURE`, `DEFAULT_TOP_P`, `DEFAULT_REPETITION_PENALTY`, `DEFAULT_NO_REPEAT_NGRAM_SIZE`
- Prompt template toggle: `DEFAULT_USE_PROMPT_TEMPLATE` (on by default)
- JA↔EN bridge: `DEFAULT_BRIDGE_JA_EN` (off by default)
- Prefix control (stabilization): `DEFAULT_PREFIX_SCALE` (0.05), `DEFAULT_PREFIX_LEN` (8), `DEFAULT_NO_PREFIX` (0 to enable)
 - Baseline automation: set `RUN_BASELINES=1` to run with and without prefix and save outputs under `outputs/`

Or call Python directly with full control:

```
python -m Player2Vec.infer_style \
  --question "川崎フロンターレの家長 昭博選手について教えて" \
  --emb checkpoints/player_embeddings.pt \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --player_id 4609 \
  --max_new_tokens 128 \
  --use_prompt_template \
  --temperature 0.3 --top_p 0.9 \
  --repetition_penalty 1.1 --no_repeat_ngram_size 4 \
  --prefix_scale 0.05 --prefix_len 8 \
  --length_penalty 1.0 --num_beams 1 --do_sample --seed 42 \
  --p_proj_ckpt checkpoints/p_proj.pt
```

## Notes

- The style LM receives the player embedding `p` (dim=256, L2-normalized) projected to the LM hidden size and prepended as a soft prefix to the question tokens.
- Default LM is Japanese-capable; the JA↔EN bridge is optional and off by default. Enable with `--bridge_ja_en` if needed.

### Stabilization & Generation Flags

- `--prefix_scale` (float, default 0.05): scales the soft prefix strength. Try 0.05→0.1.
- `--prefix_len` (int, default 8): length of the soft prefix tokens. Try 8.
- `--no_prefix`: disables prefix injection for baseline comparison.
- Suggested decoding: `--temperature 0.2~0.4`, `--top_p 0.9`, `--no_repeat_ngram_size 4`, `--repetition_penalty 1.1`.
- Additional generation controls: `--length_penalty`, `--num_beams`, `--do_sample`, `--seed`.

## Troubleshooting: Why outputs may collapse

- __Non-instruction or small LM__: Very small models can be brittle. Prefer an instruction LM such as `Qwen/Qwen2.5-1.5B-Instruct`.
- __Untrained `p_proj` mapping__: The projection `p_proj(256→hidden)` is untrained; too-strong prefixes can distort the LM embedding space.
- __Prefix injection details__: We prepend a soft prefix via `inputs_embeds` only and build a matching `attention_mask`. Mixing `input_ids` and `inputs_embeds` is avoided.

### Quick fixes (no retraining)

- Use stabilization flags: `--prefix_scale 0.05`, `--prefix_len 8`, or try `--no_prefix` for a baseline.
- Adjust decoding: lower `--temperature` to 0.2–0.4, keep `--top_p 0.9`, set `--no_repeat_ngram_size 4`, `--repetition_penalty 1.1`.
- Prefer a Japanese-capable instruction LM and disable the bridge to test native prompt following.

## Train p_proj (mini-trainer)

- Script/Module: `Player2Vec/trainer/p_proj_trainer.py`
- Input: CSV with columns `id,prompt,target` and player embeddings (`.pt/.pth/.csv`)
- Output: `checkpoints/p_proj.pt` (state_dict for `p_proj` and `prefix_ln`)

Example:

```
python -m Player2Vec.trainer.p_proj_trainer \
  --data_csv data/prefix_supervision.csv \
  --emb checkpoints/player_embeddings.pt \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --epochs 3 --batch 2 \
  --prefix_scale 0.05 --prefix_len 8 \
  --save_ckpt checkpoints/p_proj.pt
```

Load in inference via `--p_proj_ckpt checkpoints/p_proj.pt`.

## Full Pipeline Script

- `scripts/run_full_pipeline.sh`: runs
  - Player2Vec representation training (`Player2Vec/trainer/train.py`)
  - Embedding build (`scripts/build_embeddings.sh`)
  - p_proj training (`Player2Vec/trainer/p_proj_trainer.py`)

Example:

```
scripts/run_full_pipeline.sh \
  --data Data/Train \
  --feather Preprocessed_data/feather \
  --epochs 10 --batch 64 \
  --pproj_csv data/prefix_supervision.csv \
  --lm Qwen/Qwen2.5-1.5B-Instruct \
  --out checkpoints
```

For baseline comparisons during inference, you can set:

```
RUN_BASELINES=1 scripts/infer_style.sh "川崎フロンターレの家長 昭博選手について教えて"
```

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
