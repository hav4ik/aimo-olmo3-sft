# Handout: convert `chankhavu/smolmo-sft-v2-seqlen64k` → OLMo-core pre-tokenized

**You are an LLM agent on a rented server.** Goal: produce the OLMo-core pre-tokenized form of our
SFT dataset (the `.npy` token + mask arrays OLMo-core trains on), using a self-contained Docker image.
This is CPU work (no GPU). It was too slow on the owner's small box; you have many cores — but read
the hardware note, because the bottleneck is **memory bandwidth**, not cores.

## What you're producing
For each example: tokenize the conversation with the **`olmo_thinker`** chat template (Olmo-3-Think
format) and emit `token_ids_part_*.npy` (uint32) + `labels_mask_part_*.npy` (uint8, 1=train/0=masked)
+ `tokenizer/` + `dataset_statistics.json`. Sanity targets for the full run: **~2,813,055 sequences,
~37.9B tokens, trainable fraction ~0.85–0.95, max_seq_length 65536, 0 skipped.**

## Hardware: this is MEMORY-BANDWIDTH bound (read this)
Tokenization saturates memory bandwidth at ~8 cores per memory domain; extra cores then idle-spin.
So speed comes from **aggregate memory bandwidth**, not core count:
- **Prefer DDR5, multi-socket.** A CPU's bandwidth = (memory channels) × (DDR speed) × (sockets).
- **Detect sockets:** `lscpu | grep -E 'Socket|NUMA node\('` → `Socket(s): 2` and 2 NUMA nodes = dual-socket. Cloud specs usually say "2×<CPU>" or list total cores = 2× the chip's core count.
- **Among common options** (for THIS job, best→worst by bandwidth): newer **AMD EPYC Genoa/Turin (9004/9005, 12×DDR5 ch/socket)** ≫ **Xeon Platinum 8558 (Emerald Rapids, 8×DDR5-5600)** ≳ **Xeon Gold 6430 (Sapphire Rapids, 8×DDR5-4800)** ≫ **EPYC 7702 (Rome, 8×DDR4-3200 — old, weakest here despite 64 cores)**. **Dual-socket** roughly doubles all of these.
- **Use all memory domains:** on a 2-socket (2-NUMA) box, run **one shard per socket** pinned with `numactl` (see Multi-node below). One process pool spanning both sockets under-uses bandwidth due to cross-NUMA traffic.
- **RAM:** ~2 GB per worker process. **100 GB RAM is fine** if you keep `--procs ≤ ~32` per shard (which is all the bandwidth uses anyway — more procs won't speed it up and will eat RAM). 256 GB is comfortable.
- **DISK (the real requirement):** mount a **writable dir with ≥ ~450 GB** to `/out` — it holds the HF download (~35 GB), the tokenized cache (~150 GB), and the output memmaps (~190 GB). The script cleans the scratch after each shard.

## Step 1 — get the image
```bash
docker pull chankhavu/smolmo-olmocore-convert:1.0        # from Docker Hub (recommended)
# offline fallback: docker load < smolmo-olmocore-convert-1.0.tar.gz   # loads as smolmo-olmocore-convert:1.0,
#                   then: docker tag smolmo-olmocore-convert:1.0 chankhavu/smolmo-olmocore-convert:1.0
# build fallback:   docker build -t chankhavu/smolmo-olmocore-convert:1.0 .   # needs base open-instruct-dataprep:0.1.0
```

## Step 2 — HF auth (the dataset is PRIVATE)
You need an HF token with **read** access to `chankhavu/smolmo-sft-v2-seqlen64k`. Pass it as `HF_TOKEN`.

## Step 3 — run

### Single machine (simplest)
```bash
mkdir -p /data/out                       # >= 450 GB free, writable
docker run --rm \
  -e HF_TOKEN=hf_xxx \
  -e PROCS=32 \                          # ~ memory-bandwidth sweet spot; raising it won't help much
  -v /data/out:/out \
  chankhavu/smolmo-olmocore-convert:1.0
# -> /data/out/olmocore/{token_ids_part_*.npy, labels_mask_part_*.npy, tokenizer/, dataset_statistics.json}
```

### Multi-node / multi-socket (the fast path)
Each node/socket processes a slice (`--num-shards K --shard-index k`) into a **shared** `/out`.
Avoid K× downloading by pre-fetching the parquet once to shared storage and pointing `--dataset` at it:
```bash
# once, on shared storage:
hf download chankhavu/smolmo-sft-v2-seqlen64k --repo-type dataset --local-dir /shared/ds
# on node/shard k = 0..K-1 (all writing to the SAME /shared/out):
docker run --rm -e PROCS=32 -e DATASET='/out/ds/data/*.parquet' \
  -v /shared:/out chankhavu/smolmo-olmocore-convert:1.0 --num-shards K --shard-index k
# dual-socket on ONE box → run two shards pinned to each socket:
numactl --cpunodebind=0 --membind=0 docker run ... --num-shards 2 --shard-index 0 &
numactl --cpunodebind=1 --membind=1 docker run ... --num-shards 2 --shard-index 1 &
# after ALL shards finish, merge stats once:
docker run --rm -v /shared:/out chankhavu/smolmo-olmocore-convert:1.0 --combine --num-shards K
```
(`numactl` must wrap the worker; if Docker swallows it, instead launch with `--cpuset-cpus`/`--cpuset-mems` per socket.)

## Step 4 — verify before declaring success
```bash
cat /out/olmocore/dataset_statistics.json   # total_sequences ~2.81M, total_tokens ~37.9B,
                                             # trainable_fraction ~0.85-0.95, max_seq_length 65536, num_samples_skipped 0
ls /out/olmocore/token_ids_part_*.npy /out/olmocore/labels_mask_part_*.npy   # one (or more) pair(s)
```
Spot-check a sample (de-tokenize + confirm masking): read `token_ids_part_*.npy` with
`np.fromfile(path, dtype=np.uint32)` (raw memmap, NOT `np.load`) and `labels_mask_part_*.npy` with
`dtype=np.uint8`; decode with the saved `tokenizer/`. Expect: `<|im_start|>system…user…` MASKED (0),
assistant reasoning/answer/`<function_calls>` TRAINED (1), `<|im_start|>environment…` (tool outputs)
MASKED (0), the `<|im_start|>assistant\n<think>` opener MASKED, final `<|endoftext|>` TRAINED.

## Step 5 (optional) — publish back to HF
```bash
hf upload chankhavu/smolmo-sft-v2-seqlen64k /out/olmocore olmocore --repo-type dataset
```

## Troubleshooting
- **OOM during "Generating train split" (load)** — already prevented: the script writes the shard with
  small row-groups. If you still hit it, lower the shard size (raise `--num-shards`).
- **`mkdir /data: permission denied`** — you didn't mount `/out`; the cache must live under the
  writable `/out` mount (the entrypoint forces `HF_HOME=/out/.hf`).
- **Auth error / private dataset** — pass `HF_TOKEN` (read scope on `chankhavu/smolmo-sft-v2-seqlen64k`).
- **More cores didn't speed it up** — expected; it's bandwidth-bound. Shard across sockets/nodes instead.
- **Faithfulness:** this runs the exact `open-instruct convert_sft_data_for_olmocore.py --chat_template_name
  olmo_thinker --max_seq_length 65536` used for the prior dataset; the wrapper only adds HF download +
  sharding + the small-row-group fix. Verified end-to-end (trainable ~0.90, 0 skipped).
