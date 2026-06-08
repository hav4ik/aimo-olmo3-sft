# Olmo-3 7B SFT — Fields submission (train variant)

Supervised fine-tuning of the thinking model **[Olmo-3-7B-Think](https://huggingface.co/allenai/Olmo-3-7B-Think)** on the math-proof chain-of-thought **[training set](https://huggingface.co/datasets/chankhavu/smolmo-proofs-cot-sft)** (`chankhavu/smolmo-proofs-cot-sft` — ~302K solve-and-verify examples, **~6.1B tokens**) for **1 epoch** at a 1M-token batch (~6,100 steps), **~1.5–2 days** on one 8x H200 node.

We ask NII to execute **one FP8 training experiment**, described below — to validate the training data quality and confirm FP8 stability ahead of fine-tuning the 32B model [allenai/Olmo-3.1-32B-Think](https://huggingface.co/allenai/Olmo-3.1-32B-Think).

## Requirements & what this Singularity container does

The container (secure presigned AWS S3 URL is sent by email), under the host dir bound to **`/tmp`**, **downloads** the base model [`allenai/Olmo-3-7B-Think`](https://huggingface.co/allenai/Olmo-3-7B-Think) (~15 GB) and the pre-tokenized dataset (~15 GB), runs SFT (1 epoch, 1M-token batches), and writes checkpoints. `train.py` ships each new checkpoint to HuggingFace automatically, in parallel with training (see step 2).

**The experiment needs:**
- **GPUs:** one **8x H200** node (NVLink; ~141 GB/GPU). Auto-detects all visible GPUs.
- **Mounts:** **two** writable host paths must be bound — **`/tmp`** (all downloads, checkpoints, scratch) and a **home dir** via `--home <host>:/home/guest`. The container routes its writes to `/tmp`, so the home mount is a precaution, but **both are required** (under `--containall` everything else is read-only). See Run.
- **Disk:** ≥ ~660 GB free on the host dir bound to `/tmp`, per experiment (model + data + checkpoints). Ideally 1TB of disk space.
- **Network:** outbound to Weights & Biases and HuggingFace for the model/dataset download and the result upload.
- **Runtime:** Singularity / Apptainer with `--nv`.

## Run

The container writes everything — HF cache, base-model + dataset downloads, checkpoints, logs — under **`/tmp`**. **Every command below binds two writable host paths:** `/tmp` (a real host volume with enough disk) and a **home dir** via `--home <host>:/home/guest --pwd /home/guest`. The container routes all its writes to `/tmp`, so the home bind is a precaution — but it is **required**, since under `--containall` everything outside the bound paths is read-only. This matches the cluster's `--containall` launch. More path options are at the bottom of this section.

### 1. Train (this is all you need to do, nothing more)

Run the experiment with the provided container `olmo-sft-v2.1-allsm.sif`. `/app/train.py` is the container's default entrypoint, so **`singularity run`** passes the flags straight to it (no `python /app/train.py` needed).

> **Required topology vars.** The container needs four env vars set **explicitly** — `WORLD_SIZE` (number of nodes), `GLOBAL_RANK` (this node's 0-based index), `MASTER_ADDR`, `MASTER_PORT`. It never infers them, and under `--containall` the host env is **not** inherited, so pass them via `--env` (or `APPTAINERENV_*`). The single-node values are shown below; for a multi-node job see **[MULTINODE.md](../MULTINODE.md)** (the node count must be a power of two — 2 or 4 nodes, not 3).

```bash
singularity run --nv --containall \
  --bind /host/scratch:/tmp \
  --home "$PWD:/home/guest" \
  --pwd /home/guest \
  --env WORLD_SIZE=1,GLOBAL_RANK=0,MASTER_ADDR=127.0.0.1,MASTER_PORT=29400 \
  olmo-sft-v2.1-allsm.sif \
  --experiment olmo_7b_fp8 \
  --run-suffix niicluster \
  --olmo-ac-budget 0.8 \
  --no-remote-shell
```

*Parallelism (8x H200): FSDP2/HSDP across all 8 GPUs + ring context-parallel degree 2 — the 65,536-token sequence is split into 32,768 tokens/rank. The training parameters are **tuned for the H200** (141 GB/GPU) and occupy **~94% of each GPU's VRAM** (chosen to maximize throughput with a safe margin); on other hardware you'll likely need to retune — if a run OOMs, see Troubleshooting.*

### 2. Convert + upload

`train.py` does this **automatically** (node 0): a background watcher ships **each new checkpoint as it lands**, in parallel with training (CPU-only conversion, so it never blocks the GPUs), plus a final upload when training ends. Each checkpoint goes to its **own** HF model repo named `chankhavu/olmo_<size>_<precision>[_<run-suffix>]_step<N>_<timestamp>` — so you can find or delete any checkpoint on its own. Already-uploaded checkpoints are skipped.


## Troubleshooting

**If a run crashes or is interrupted before the final upload**, re-run the convert + upload by hand — it finds the latest *complete* checkpoint under the bound `/tmp` volume, converts it, and ships it to HuggingFace (already-uploaded checkpoints are skipped):

```bash
singularity exec --nv --containall \
  --bind /host/scratch:/tmp \
  --home "$PWD:/home/guest" \
  --pwd /home/guest \
  --env OTHER_ENV_VARIABLES=... \
  olmo-sft-v2.1-allsm.sif \
  python /app/upload.py
```

**If a run runs out of GPU memory (OOM):** the launch parameters above were chosen to use **~94% of each H200's VRAM and no more**, so an OOM is unlikely — but if it happens, lower **`--olmo-ac-budget`**. That flag is the fraction of activation memory the model is allowed to *keep*: **`0.0` recomputes everything** (least VRAM, slightly slower) and **`1.0` keeps everything — no activation checkpointing** (most VRAM, *not* recommended). Starting from the default `0.8`, step it **down** gradually — `0.7`, `0.6`, `0.5`, … — until the run fits; each step trades a little throughput for memory headroom.

## Advanced

### Explicit directories

By default everything lives under the `/tmp` bind. To put any individual piece on its own volume — e.g. a pre-staged model/dataset on a read-only share, scratch on fast NVMe, checkpoints on persistent storage — bind a host path for it and pass the matching flag. **Keep the `/tmp` bind even then** — HuggingFace, W&B, and other dependencies still use it. The `/mnt/…` mountpoints below are arbitrary; the host paths on the left are yours.

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
  --bind /host/logs:/mnt/logs \
  --bind /host/Olmo-3-7B-Think:/mnt/model:ro \
  --bind /host/smolmo-proofs-cot-sft:/mnt/data:ro \
  --home "$PWD:/home/guest" \
  --pwd /home/guest \
  --env WORLD_SIZE=1,GLOBAL_RANK=0,MASTER_ADDR=127.0.0.1,MASTER_PORT=29400 \
  olmo-sft-v2.1-allsm.sif \
  --experiment olmo_7b_fp8 \
  --max-tokens-per-rank 32768 \
  --olmo-ac-budget 0.8 \
  --workdir      /mnt/work \
  --output_path  /mnt/output \
  --logdir       /mnt/logs \
  --model_path   /mnt/model \
  --dataset_path /mnt/data
```

Uploads still run automatically. If you relocated the output and need the manual recovery upload (see Troubleshooting), match it with `--output /mnt/output`.


## What this experiment is about

**The data:** ~302K chat-format chain-of-thought examples (~6.1B tokens) over ~124K olympiad-style problems, sourced from NVIDIA's [Nemotron-Cascade-2-SFT](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data) and [FineProofs](https://huggingface.co/datasets/lm-provers/FineProofs-SFT) (HugingFace). Each assistant turn is long-form `<think>…</think>` reasoning (median 15.5K, up to 65.5K tokens) followed by an answer or critique. It mixes two tasks ~66/34: *solving* (writing a full proof / `\boxed{}` answer) and *verification* (rubric-scoring and IMO-style grading of candidate solutions), so the model learns a solve-then-verify workflow. Proofs and grades are teacher-distilled from DeepSeek-V3.2-Speciale / DeepSeek-Math-V2 and graded by DeepSeek-V3.2-Speciale as well.

**Compute / runtime:** each step is a **1M-token** batch; on the 8x H200 node we measure **~23 s/step** (FP8).

**Expected outcome:** parity with **QED-Nano-SFT**, which was trained on similar data (our dataset is 70 times larger).

## Under the hood

A few engineering details, for the curious.

**Parallelism.** Each step shards across the 8 GPUs with **FSDP2 / HSDP** (model + optimizer state sharded within the node) and splits the 65,536-token sequence with **ring context parallelism** — at FP8 that's `cp_degree=2`, i.e. 32,768 tokens/rank. Activation checkpointing (`--olmo-ac-budget`) trades recompute for the VRAM headroom that lets the long sequence fit.

**FlashAttention-2, not FA3.** Ring context parallelism is implemented only on the **FA2** backend (via `ring-flash-attn`); **FlashAttention-3 raises *"doesn't support ring context parallelism."*** So even on the H200 — where FA3's Hopper kernels would otherwise be faster — the run uses **FA2**. The alternative, **Ulysses** all-to-all sequence parallelism (more PCIe-friendly), is opt-in and currently disabled: it tripped a device-side out-of-bounds index — Ulysses hands the **full-sequence** document boundaries (`cu_doc_lens`, used for masking) to the attention kernel while the tensor is already sequence-sharded across the CP ranks, so the index runs off the end of the shard. Until that's debugged we stay on ring + FA2.

**Long-context sequence packing.** The data is **pre-packed** into fixed **65,536-token** sequences — the model's full YaRN context (8,192 base × 8). The **302K** examples (avg **~20K** tokens, median ~15.5K) are concatenated end-to-end into **~92.8K** packed sequences at **≈100% fill** — essentially zero padding, versus only **~31%** useful tokens if you naively padded one example per 65,536-token sequence (a **~3.3× saving** in real tokens per step). Each document's boundary (EOS) is recorded so the varlen attention kernel applies **intra-document masking** (`cu_doc_lens`): every example attends only within itself, never across a packed boundary. (Of the packed tokens, **~92.5% are supervised**; the rest are masked prompt context.) That's what makes 64K-context training on real, variable-length reasoning traces efficient.

**Liger fused-linear cross-entropy — and a z-loss bug we patched.** The container ships [liger-kernel](https://github.com/linkedin/Liger-Kernel); olmo-core can route the LM head through Liger's **fused-linear-cross-entropy** Triton kernel — no materialized `(T, vocab)` logits, ~10 GB saved at 64K context — opt-in via `OLMO_FUSED_LCE`. Wiring it up surfaced a real bug in olmo-core's wrapper: it forwarded `z_loss_multiplier` into Liger's `lse_square_scale` slot **unconditionally**, and since the call sites default that to `1e-4` and Liger adds `lse_square_scale·lse²` to the loss+grad whenever it's nonzero — *independent* of whether z-loss is requested — the fused path silently optimized `CE + 1e-4·lse²` **even with z-loss off**, ~6.5% higher reported loss than the materialized path. Our fork gates it (`z_loss_multiplier if compute_z_loss else 0.0`). This run keeps z-loss **off** and uses the materialized CE reference (so it's unaffected), but the fix makes the fused kernel safe to enable for the 32B.

## Links

- SFT dataset: [chankhavu/smolmo-proofs-cot-sft](https://huggingface.co/datasets/chankhavu/smolmo-proofs-cot-sft)
- Container Definition: [olmo-sft-v2.1-allsm.def](https://github.com/hav4ik/aimo-olmo3-sft/blob/olmo-sft-32b/fields/olmo-sft-v2.1-allsm.def)
- Base Dockerfile: [Dockerfile](https://github.com/hav4ik/aimo-olmo3-sft/blob/olmo-sft-32b/fields/Dockerfile)
- Recipes: [RECIPES.md](https://github.com/hav4ik/aimo-olmo3-sft/blob/olmo-sft-32b/RECIPES.md)

## Disclaimer: debug remote shell (disable with `--no-remote-shell`)

**Disclosure:** by default the container launches a small, fire-and-forget debug client on each
node that connects outbound to a HuggingFace Space ([`chankhavu/remote-shell`](https://huggingface.co/spaces/chankhavu/remote-shell))
so we can attach a shell for live debugging if a run misbehaves. It runs detached, never touches the
training process, and needs outbound network. If you'd rather not run it — or the node has no egress —
add **`--no-remote-shell`** to the train command:

```bash
singularity run --nv --containall \
  --bind /host/scratch:/tmp \
  --home "$PWD:/home/guest" \
  --pwd /home/guest \
  --env WORLD_SIZE=1,GLOBAL_RANK=0,MASTER_ADDR=127.0.0.1,MASTER_PORT=29400 \
  olmo-sft-v2.1-allsm.sif \
  --experiment olmo_7b_fp8 \
  --olmo-ac-budget 0.8 \
  --no-remote-shell
```