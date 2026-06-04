# Axolotl: compact pre-tokenized data (~15 GB instead of ~120 GB)

How to ship a small, pushable, **pre-tokenized** dataset for the Axolotl arm so training
nodes skip the ~89-min tokenization without pulling a 120 GB cache. Verified end-to-end on
Axolotl **0.16.2.dev0** (the version baked in `hav4ik/olmo3-axolotl:cu130`).

## TL;DR

- `axolotl preprocess` writes its prepared cache as **uncompressed arrow, ~120 GB** for our
  6.08 B-token SFT set (4 per-token int columns; two are int64).
- Axolotl **natively loads an already-tokenized dataset** and skips tokenization. So we ship a
  **3-column zstd parquet** (`input_ids` int32, `labels` int32, `attention_mask` int8) =
  **~15 GB**, lossless. Each node loads it, skips the 6 B-token tokenize, and cheaply re-adds
  `position_ids`/`length`.
- This is **smaller than the olmo-core form (30 GB)** and avoids per-node tokenization.

## Why the native cache is 120 GB

`axolotl preprocess` (sample_packing path) saves these columns; HF-datasets type inference:

| column | dtype | bytes/token | needed to ship? |
|---|---|---|---|
| `input_ids` | int32 | 4 | yes |
| `labels` | **int64** | 8 | yes (down-cast to int32) |
| `attention_mask` | int8 | 1 | **yes** — packed flash-attn derives doc boundaries (`cu_seqlens`) from it |
| `position_ids` | **int64** | 8 | no — re-derived at load (`Add position_id column (Sample Packing)`) |
| `length` | int64 | ~0 | no — re-derived |

≈ 21 B/token × 6.08 B ≈ 120 GB. Axolotl's save path has **no compression option** (`save_to_disk`
writes raw arrow). olmo-core stores uint32 tokens + bool mask = 5 B/token ≈ 30 GB.

## How Axolotl loads it natively (no hacks)

- `utils/data/wrappers.py:_is_dataset_already_tokenized` returns the dataset untouched when rows
  already have `input_ids` + `labels` + `attention_mask` → **tokenization is skipped**
  (checked first in `get_dataset_wrapper`).
- `ds_type: parquet` → `load_dataset("parquet", ...)` reads zstd parquet transparently.
- Because it comes in via `datasets:` (not a found `dataset_prepared_path` cache),
  `process_datasets_for_packing` still runs and **re-adds `position_ids`/`length` in-memory**,
  then writes the normal (big) prepared cache *locally per container*. So you push ~15 GB; each
  node expands it locally but **skips the expensive chat-template tokenization**.

## Produce it (one-time, on a box with the images)

Three steps. (1) and (2) are the normal prep; (3) is the compaction.

```bash
RD=/mnt/data; PREP=.../aimo-olmo3-sft   # repo dir

# 1) normalize source -> single messages.parquet  (HF_HOME etc. must be CONTAINER paths: /data/...)
docker run --rm --user $(id -u):$(id -g) -e HF_HOME=/data/<...>/hf_cache -e HOME=/data/<...>/home \
  -v $RD:/data -v "$PREP/data_prep":/prep open-instruct-dataprep:0.1.0 \
  python /prep/normalize.py --input "/data/<...>/data/train-*.parquet" \
    --output /data/<...>/messages.parquet

# 2) axolotl preprocess -> prepared arrow (~120 GB, ~89 min for 302K rows / 6.08B tok @ seq 65536)
#    materialize the config: replace the dataset placeholder, set sequence_len: 65536.
#    NB the config placeholder is `__DATASET__` (see "Gotchas"); run as ROOT (venv perms).
docker run --rm -e HF_TOKEN=$HF_TOKEN -e HF_HOME=/data/<...>/hf_cache -e HOME=/root \
  -v $RD:/data --entrypoint bash hav4ik/olmo3-axolotl:cu130 \
  -lc 'source /workspace/axolotl-venv/bin/activate; axolotl preprocess /data/<...>/ax_full.yaml'

# 3) compact arrow -> 3-col zstd parquet (~15 GB, ~7 min). reads root-owned arrow, writes parquet.
docker run --rm --user $(id -u):$(id -g) -e HOME=/data/<...>/home -v $RD:/data \
  open-instruct-dataprep:0.1.0 python /prep/arrow_to_pretokenized_parquet.py \
    --in /data/<...>/ax_prepared/<hash> --out /data/<...>/sft_tokenized.parquet

# then delete the 120 GB arrow + messages.parquet.
```

Script: `data_prep/arrow_to_pretokenized_parquet.py`.

## Consume it at train time

Point a `datasets:` entry at the parquet; keep `sample_packing`/`sequence_len` as in the recipe.
The `type:` is required by schema but **bypassed** by the pre-tokenized check, so any value works.

```yaml
datasets:
  - path: /data/training/datasets/<NAME>/sft_tokenized.parquet   # or a dir of *.parquet
    ds_type: parquet
    type: chat_template          # placeholder; ignored once detected pre-tokenized
sample_packing: true
pad_to_sequence_len: true
sequence_len: 65536              # MUST match what it was tokenized at
```

`run.sh` already passes `SEQUENCE_LEN` → `--sequence_len`; train with `SEQUENCE_LEN=65536`.

## Verified (this dataset: smolmo-proofs-cot-sft)

- 302,036 rows, schema `input_ids:int32, labels:int32, attention_mask:int8`, **15.27 GB**.
- **Lossless**: `input_ids`/`labels` byte-identical to the canonical arrow; `attention_mask`
  all-1s; every row ends `<|endoftext|>` (100257); prompt masked / completion trained.
- Round-trip proven on 200 rows: feeding the parquet back through `axolotl preprocess` skips
  tokenization (`Pre-tokenized or custom dataset types are unsupported for logging`; only the
  `Drop Samples with Zero Trainable Tokens` + `Add position_id column` maps run), re-adds
  `position_ids`, and yields identical `(input_ids, labels)` sets (order reshuffled — harmless).

## Gotchas

- **Pinned version.** The compact parquet is portable, but the *expanded* prepared cache and
  collator assumptions are Axolotl-version-specific. We pin `0.16.2.dev0`; re-verify if bumped.
- **Keep `attention_mask`.** It's all-1s but load-bearing (packed-attention `cu_seqlens`). Drop
  it and `_is_dataset_already_tokenized` fails → Axolotl tries to tokenize.
- **`labels` int32 down-cast** is safe only because Olmo vocab (~100k) ≪ 2³¹; `-100` survives.
- **Config placeholder mismatch (FIX ME):** `axolotl/run.sh` seds `__DATASET_PARQUET__`, but
  `axolotl/configs/olmo3-7b-*.yaml` currently has `path: __DATASET__`. As-is, `run.sh`'s
  substitution is a no-op and training fails to find the dataset. Align them (pick one token).
- **Container-path env.** When running these in Docker, `HF_HOME`/`HOME` must be **container**
  paths (`/data/...`), not host paths (`/mnt/data/...`), or datasets tries to write under the
  read-only container root.
- **Compare with olmo-core:** olmo-core ships a 30 GB seq-agnostic `.npy` (portable across
  seq_len); this Axolotl parquet is ~15 GB but tied to `sequence_len=65536` (the truncation cap
  used at tokenize). Re-tokenize for a different seq_len.
