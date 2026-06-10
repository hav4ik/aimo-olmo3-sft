# Olmo-3 32B SFT — Fields submission (train variant)

Supervised fine-tuning of the thinking model **[Olmo-3.1-32B-Think](https://huggingface.co/allenai/Olmo-3.1-32B-Think)** on the math-proof / numeric / tool-use chain-of-thought **[training set](https://huggingface.co/datasets/chankhavu/smolmo-sft-v2-seqlen64k)** (`chankhavu/smolmo-sft-v2-seqlen64k` — **2,813,055 examples, ~37.9 B tokens**, packed to 64K-token sequences) at a **1.5M-token batch**, on one or more **8x H200** nodes.

This follows the 7B FP8 experiment: it scales the validated pipeline up to the 32B model. The container is the same family (v2.2), built and verified identically to the 7B's v2.1 image, but with changes for better stability and logging.

## Requirements & what this Singularity container does

The container (secure presigned AWS S3 URL is sent by email), under the host dir bound to **`/tmp`**, **downloads** the base model [`allenai/Olmo-3.1-32B-Think`](https://huggingface.co/allenai/Olmo-3.1-32B-Think) (~64 GB) and the **pre-tokenized** olmo-core dataset [`chankhavu/smolmo-sft-olmocore-pretokenized`](https://huggingface.co/datasets/chankhavu/smolmo-sft-olmocore-pretokenized) (~177 GB), converts the base model to olmo-core distcp once, runs SFT, and writes checkpoints. `train.py` ships each new checkpoint to HuggingFace automatically, in parallel with training (see step 2).

**The experiment needs:**
- **GPUs:** Ideally **3 or 4 nodes with 8x H200** (NVLink; ~141 GB/GPU). The 32B also runs across **2, 3, 4, or 6 nodes** (the 1.5M-token batch divides evenly for those counts) — see [MULTINODE.md](../MULTINODE.md).
- **Mounts:** **two** writable host paths must be bound — **`/tmp`** (all downloads, checkpoints, scratch) and a **home dir** via `--home <host>:/home/guest`. The container routes its writes to `/tmp`, so the home mount is a precaution, but **both are required** (under `--containall` everything else is read-only). See Run.
- **Disk:** **≥ ~1.3 TB free** (ideally **1.5 TB**) on the host dir bound to `/tmp`. The 32B is larger than the 7B: base download (~64 GB) + distcp conversion (~121 GB) + dataset (~177 GB) + training checkpoints (~251 GB each; `--keep-last 1` keeps one persistent + one rotating ephemeral). If the node is capped at ~1 TB, contact us — we have a checkpoint configuration that fits.
- **Network:** outbound to Weights & Biases and HuggingFace for the model/dataset download and the result upload (the 64 GB checkpoint upload assumes a decent uplink; bump `FIELDS_UPLOAD_TIMEOUT` from the 90-min default if it's slow).
- **Runtime:** Singularity / Apptainer with `--nv`.

## Run

The container writes everything — HF cache, base-model + dataset downloads, checkpoints, logs — under **`/tmp`**. **Every command below binds two writable host paths:** `/tmp` (a real host volume with enough disk) and a **home dir** via `--home <host>:/home/guest --pwd /home/guest`. The container routes all its writes to `/tmp`, so the home bind is a precaution — but it is **required**, since under `--containall` everything outside the bound paths is read-only. This matches the cluster's `--containall` launch. More path options are at the bottom of this section.

### 1. Train (this is all you need to do, nothing more)

Run the experiment with the provided container `olmo-sft-v2.2-allsm.sif`. `/app/train.py` is the container's default entrypoint, so **`singularity run`** passes the flags straight to it. **A bare run trains the 32B at its fitting shape** — the defaults (`--experiment olmo_32b_fp8`, `--max-tokens-per-rank 16384`, `--olmo-ac-budget 0.4`, 1.5M-token batch, `--keep-last 1`) are already the 32B's, so you don't pass them.

> **Required topology vars.** The container needs four env vars set **explicitly** — `WORLD_SIZE` (number of nodes), `GLOBAL_RANK` (this node's 0-based index), `MASTER_ADDR`, `MASTER_PORT`. It never infers them, and under `--containall` the host env is **not** inherited, so pass them via `--env` (or `APPTAINERENV_*`). The single-node values are shown below; for multi-node (2/3/4/6 nodes) see **[MULTINODE.md](../MULTINODE.md)**.

```bash
singularity run --nv --containall \
  --bind /host/scratch:/tmp \
  --home "$PWD:/home/guest" \
  --pwd /home/guest \
  --env WORLD_SIZE=1,GLOBAL_RANK=0,MASTER_ADDR=127.0.0.1,MASTER_PORT=29400 \
  olmo-sft-v2.2-allsm.sif \
  --run-suffix niicluster
```

*Parallelism (8x H200): FSDP2/HSDP across all 8 GPUs + ring context-parallel degree 4 — the 65,536-token sequence is split into 16,384 tokens/rank. Tuned for the H200 (141 GB/GPU); at `--olmo-ac-budget 0.4` it occupies **~85% of each GPU's VRAM (~118 GB)** with a safe margin. If a run OOMs, see Troubleshooting.*

*Learning rate: peak **5e-5**, **cosine** schedule with a **0.1 floor** (decays to 5e-6) and 3% warmup.*

### 2. Convert + upload

`train.py` does this **automatically** (node 0): a background watcher ships **each new checkpoint as it lands**, in parallel with training (CPU-only conversion, so it never blocks the GPUs), plus a final upload when training ends. Each checkpoint goes to its **own** HF model repo named `chankhavu/olmo_32b_fp8[_<run-suffix>]_step<N>_<timestamp>` — so you can find or delete any checkpoint on its own. Already-uploaded checkpoints are skipped.

## Troubleshooting

**If a run crashes or is interrupted before the final upload**, re-run the convert + upload by hand — it finds the latest *complete* checkpoint under the bound `/tmp` volume, converts it, and ships it to HuggingFace (already-uploaded checkpoints are skipped):

```bash
singularity exec --nv --containall \
  --bind /host/scratch:/tmp \
  --home "$PWD:/home/guest" \
  --pwd /home/guest \
  olmo-sft-v2.2-allsm.sif \
  python /app/upload.py
```

**If a run runs out of GPU memory (OOM):** the default `--olmo-ac-budget 0.4` uses ~85% of each H200's VRAM, so an OOM is unlikely — but if it happens, lower **`--olmo-ac-budget`**. That flag is the fraction of activation memory the model is allowed to *keep*: **`0.0` recomputes everything** (least VRAM, slightly slower) and **`1.0` keeps everything** (most VRAM). Step it **down** — `0.3`, `0.2`, … `0.0` — until the run fits; each step trades a little throughput for memory headroom.

## Advanced

### Explicit directories

By default everything lives under the `/tmp` bind. To put any individual piece on its own volume — e.g. a pre-staged model/dataset on a read-only share, scratch on fast NVMe, checkpoints on persistent storage — bind a host path for it and pass the matching flag. **Keep the `/tmp` bind even then** — HuggingFace, W&B, and other dependencies still use it.

| flag | relocates | default |
|---|---|---|
| `--workdir`      | downloads, HF cache, W&B, compile caches, scratch | `/tmp/olmo-sft/work` |
| `--output_path`  | checkpoints | `/tmp/olmo-sft/output` |
| `--logdir`       | logs | `<output>/logs` |
| `--model_path`   | an **existing** base-model dir (skips the model download) | downloaded into the work dir |
| `--dataset_path` | an **existing** tokenized-dataset dir (skips the data download) | downloaded into the work dir |

```bash
singularity run --nv --containall \
  --bind /host/scratch:/tmp \
  --bind /host/fast:/mnt/work \
  --bind /host/persistent:/mnt/output \
  --bind /host/Olmo-3.1-32B-Think:/mnt/model:ro \
  --bind /host/smolmo-sft-olmocore-pretokenized:/mnt/data:ro \
  --home "$PWD:/home/guest" \
  --pwd /home/guest \
  --env WORLD_SIZE=1,GLOBAL_RANK=0,MASTER_ADDR=127.0.0.1,MASTER_PORT=29400 \
  olmo-sft-v2.2-allsm.sif \
  --workdir      /mnt/work \
  --output_path  /mnt/output \
  --model_path   /mnt/model \
  --dataset_path /mnt/data
```

Uploads still run automatically. If you relocated the output and need the manual recovery upload (see Troubleshooting), match it with `--output /mnt/output`.

## What this experiment is about

**The data** ([`chankhavu/smolmo-sft-v2-seqlen64k`](https://huggingface.co/datasets/chankhavu/smolmo-sft-v2-seqlen64k)): **2,813,055** chat-format chain-of-thought examples, **~37.9 B tokens**, sourced from NVIDIA's Nemotron datasets and FineProofs, teacher-distilled from DeepSeek-V3.x / V4. Each assistant turn is long-form `<think>…</think>` reasoning followed by an answer, critique, or tool call. It blends three task families:

| Task family | Tokens | Examples |
|---|---|---|
| **Proofs** (math_proof, proofs_v2, fineproofs) | ~16 B | 832,233 |
| **Numeric-answer** (math_notool, math_v4_cot, math_v4_tir_nc) | ~11 B | 747,740 |
| **Tool use** (math_withtool, math_v4_tir — `stateful_python_code_exec`) | ~11 B | 1,133,142 |

So the model learns to *solve* (full proof / `\boxed{}` answer), *verify*, and *use tools* (Python code execution).

**Batch / runtime:** each step is a **1.5M-token** batch (1.5M = 1,572,864). At 64K context the model runs FP8 across the 8 H200s. *(Step time and total run length depend on the AC budget, node count, and the step/epoch target — confirm against the live run before quoting NII a duration.)*

## Under the hood

A few engineering details, for the curious.

**Parallelism.** Each step shards across the 8 GPUs with **FSDP2 / HSDP** (model + optimizer state sharded within the node) and splits the 65,536-token sequence with **ring context parallelism** — at this shape `cp_degree=4`, i.e. **16,384 tokens/rank**. Across 2/3/4/6 nodes the model still shards within each node and replicates across nodes (HSDP), so per-GPU memory is unchanged; only the data-parallel width (and grad-accum) changes. Activation checkpointing (`--olmo-ac-budget 0.4`) trades a little recompute for the VRAM headroom that lets the long sequence fit.

**FlashAttention-2, not FA3.** Ring context parallelism is implemented only on the **FA2** backend (via `ring-flash-attn`); FlashAttention-3 doesn't support ring CP. So even on the H200 the run uses **FA2**.

**Long-context sequence packing.** The data is **pre-packed** into fixed **65,536-token** sequences — the model's full YaRN context (8,192 base × 8) — in the pre-tokenized olmo-core dataset. The ~2.8M variable-length examples are concatenated end-to-end at ≈100% fill (essentially zero padding). Each document's boundary (EOS) is recorded so the varlen attention kernel applies **intra-document masking** (`cu_doc_lens`): every example attends only within itself, never across a packed boundary.

**Liger fused-linear cross-entropy.** The container ships [liger-kernel](https://github.com/linkedin/Liger-Kernel); olmo-core routes the LM head through Liger's **fused-linear-cross-entropy** Triton kernel (no materialized `(T, vocab)` logits, ~10 GB saved at 64K context), enabled by default (`OLMO_FUSED_LCE=1`). Our fork patches an olmo-core z-loss bug (it forwarded `z_loss_multiplier` into Liger's `lse_square_scale` slot unconditionally, silently optimizing `CE + 1e-4·lse²` even with z-loss off); the fix gates it (`z_loss_multiplier if compute_z_loss else 0.0`) so the fused kernel is correct.

## Links

- Training set (raw): [chankhavu/smolmo-sft-v2-seqlen64k](https://huggingface.co/datasets/chankhavu/smolmo-sft-v2-seqlen64k) — 2,813,055 examples, ~37.9 B tokens
- Pre-tokenized (olmo-core) dataset the container downloads: [chankhavu/smolmo-sft-olmocore-pretokenized](https://huggingface.co/datasets/chankhavu/smolmo-sft-olmocore-pretokenized)
- Base model: [allenai/Olmo-3.1-32B-Think](https://huggingface.co/allenai/Olmo-3.1-32B-Think)
- Container definition: [olmo-sft-v2.2-allsm.def](https://github.com/hav4ik/aimo-olmo3-sft/blob/olmo-sft-32b/fields/olmo-sft-v2.2-allsm.def)
- Multi-node guide: [MULTINODE.md](../MULTINODE.md)
</content>
