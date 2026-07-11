# Fast OLMo-core conversion — optimal redesign

Goal: convert `chankhavu/smolmo-sft-v2-seqlen64k` (1024 parquet shards, ~37.9B tokens) to the
OLMo-core pretokenized memmap form **byte-identically** to the stock open-instruct
`convert_sft_data_for_olmocore.py --chat_template_name olmo_thinker`, but ~3× faster by removing
two single-threaded bottlenecks and the redundant re-write/re-load.

## Baseline (what the stock wrapper path does, per shard)
1. load full dataset → Arrow cache (~113 GB)
2. **re-write** to one `messages.parquet` with small row-groups (single-thread, ~15 min)  ← redundant
3. converter loads that parquet again → second Arrow cache
4. `Map(num_proc=N)` tokenize  (parallel — the only step that should dominate)
5. `Filter(num_proc=N)` drop > max_seq_length
6. **`count_tokens` stats `Map` (NO num_proc)** — single-thread, ~1.5 h/part  ← pure waste
7. `Collecting tokens` write loop (single-thread) → `.bin` → chunk to `.npy`

Steps 2 and 6 are serial and add hours; step 1/3 double-cache.

## Key facts established (see validate/RESULTS.md)
- **The 1024 shards are already globally shuffled** (bucket-shuffle seed 20260609): source mix is
  uniform across shard 0/512/1023. No reshuffle needed.
- **The 1024 shards already have small row-groups** (3000 rows / 38 MB / 1 row-group each) — the
  exact layout the `messages.parquet` re-write was creating. The re-write only existed to fix the
  OLD 115 GB-single-file-giant-row-group OOM, which does not apply here.
- `labels_mask = (labels != -100)`; tokenization + masking happen entirely in the
  `Map(num_proc=N)` step (open-instruct transform fns). `count_tokens` only reads them for stats.

## The redesign — two surgical patches to `dataset_transformation.py` (tokenization untouched)
1. **`count_tokens` → add `num_proc=max_num_processes()`** (patch hunk 3). Parallelizes stats only;
   does not read/write a single output token. PROVEN byte-identical (sha256 match, 0 diffs).
2. **parquet dispatch accepts a glob OR comma-separated file list** (patch hunk 1+2). `load_dataset`
   already supports `data_files=<glob|list>`; the only blocker was the `os.path.exists()` guard.
   Lets us feed the 1024 shards directly — **no re-write, no double-load**.

## Sharding across nodes — by FILE SUBSET (not `.shard()`)
Because every shard is independently shuffled, assign each node a contiguous slice of the 1024
files (node k of K → files `[k*1024/K : (k+1)*1024/K]`), passed as a comma-separated list to
`--dataset_mixer_list`. Each node loads ONLY its files (no full-dataset load per node). Output part
files are prefixed per node so OLMo-core's `token_ids_part_*.npy` glob reads them all together.

### Ordering decision (user, 2026-06): DETERMINISTIC, not seed-based
- **Single node (current):** one process over all 1024 shards → exactly the stock single-process path
  (one `shuffle(seed=42)`). No equivalence question.
- **Multi-node (future):** do NOT rely on `shuffle(seed=42)` matching across nodes (different
  `datasets`/numpy/RNG impls could drift). Instead make on-disk order a DETERMINISTIC function of the
  file subset: drop the converter's redundant internal `shuffle` (numpy_dataset_conversion.py:256) —
  the 1024 shards are already globally bucket-shuffled at build time, and olmo-core reshuffles
  documents at train time, so the internal shuffle adds nothing. Removing it is an ORDER-only change
  (tokens/masks untouched) and makes every node's output reproducible byte-for-byte from its inputs.
  (Not implemented yet; single-node path needs none of this.)

## What is removed vs. baseline
- the `messages.parquet` re-write (step 2) — gone (feed shards directly)
- the full-dataset load-per-shard (step 1/3) — each node loads only its file subset
- the 1.5 h serial `count_tokens` (step 6) — parallelized
- the double Arrow cache — single load per node

## Safety / fidelity argument
The two patches change (a) the parallelism of a stats-only counter and (b) how input files are
*read*. Neither alters the transform functions that produce `input_ids`/`labels`. Therefore output
tokens + masks are identical to baseline. This is asserted, not assumed: validate/ runs the stock
(unpatched) converter and the patched converter on the same inputs and diffs the `.npy` bytes.
