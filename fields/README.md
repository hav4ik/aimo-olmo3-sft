# Olmo-3 7B SFT — Fields submission (train variant)

Supervised fine-tuning of **Olmo-3 7B (Think)** on math-proof chain-of-thought data ([`chankhavu/smolmo-proofs-cot-sft`](https://huggingface.co/datasets/chankhavu/smolmo-proofs-cot-sft) — ~302K solve-and-verify examples, **~6.1B tokens**) for **1 epoch** at a 1M-token batch (~6,100 steps), **~1.5–2 days** on one 8x H200 node.

We ask NII to execute 2 experiments, described below: FP8 and BF16 training. The purpose is to (A) validate data quality and (B) compare stability of FP8 vs BF16. This will help us determine the best training approach for the 32B model [allenai/Olmo-3.1-32B-Think](https://huggingface.co/allenai/Olmo-3.1-32B-Think).

## Requirements & what it does

For each experiment the container, under the host dir bound to **`/tmp`**, **downloads** the base model
[`allenai/Olmo-3-7B-Think`](https://huggingface.co/allenai/Olmo-3-7B-Think) (~15 GB) and the pre-tokenized dataset (~15 GB), runs SFT (1 epoch, 1M-token batches), and writes checkpoints. `train.py` ships each new checkpoint to HuggingFace automatically, in parallel with training (see step 2).

**Each of the 2 experiments needs:**
- **GPUs:** one **8x H200** node (NVLink; ~141 GB/GPU). Auto-detects all visible GPUs.
- **Disk:** ≥ ~660 GB free on the host dir bound to `/tmp`, per experiment (model + data + checkpoints). Ideally 1TB of disk space.
- **Network:** outbound to Weights & Biases and HuggingFace for the model/dataset download and the result upload.
- **Runtime:** Singularity / Apptainer with `--nv`.

## Run

The container writes everything — HF cache, base-model + dataset downloads, checkpoints, logs — under **`/tmp`** by default. **Bind `/tmp` to a real host volume** and nothing else needs to be specified; this matches the cluster's `--containall` launch (where only explicitly-bound paths are writable). More path options are at the bottom of this section.

### 1. Train

Run each experiment with the provided container `olmo3-fields_cu130.sif`. `/app/train.py` is the container's default entrypoint, so `singularity run --containall … olmo3-fields_cu130.sif --experiment …` runs it directly; the explicit `singularity exec … python /app/train.py …` form below is equivalent and just makes the entrypoint visible.

#### Experiment 1: olmo_7b_fp8

```bash
singularity exec --nv --containall \
  --bind /host/scratch:/tmp \
  olmo3-fields_cu130.sif \
  python /app/train.py \
  --experiment olmo_7b_fp8 \
  --max-tokens-per-rank 32768 \
  --olmo-ac-budget 0.8
```

*Parallelism (8x H200): FSDP2/HSDP across all 8 GPUs + ring context-parallel degree 2 — the 65,536-token sequence is split into 32,768 tokens/rank.*

#### Experiment 2: olmo_7b_bf16

```bash
singularity exec --nv --containall \
  --bind /host/scratch:/tmp \
  olmo3-fields_cu130.sif \
  python /app/train.py \
  --experiment olmo_7b_bf16 \
  --max-tokens-per-rank 16384 \
  --olmo-ac-budget 1.0
```

*Parallelism (8x H200): FSDP2/HSDP across all 8 GPUs + ring context-parallel degree 4 — the 65,536-token sequence is split into 16,384 tokens/rank.*

### 2. Convert + upload

`train.py` does this **automatically** (node 0): a background watcher ships **each new checkpoint as it lands**, in parallel with training (CPU-only conversion, so it never blocks the GPUs), plus a final upload when training ends. Each checkpoint goes to its **own** HF model repo named `chankhavu/olmo_<size>_<precision>[_<run-suffix>]_step<N>_<timestamp>` — so you can find or delete any checkpoint on its own. Already-uploaded checkpoints are skipped.

To (re)run a convert+upload by hand (e.g. after a crash):

```bash
singularity exec --nv --containall \
  --bind /host/scratch:/tmp \
  olmo3-fields_cu130.sif \
  python /app/upload.py
```

Tune the watcher poll interval with `FIELDS_UPLOAD_WATCH_INTERVAL` (seconds, default 300); disable all uploads with `--no-upload`.

### Optional: explicit directories

If you'd rather place the **work dir** (downloads, HF cache, scratch) and the **output dir** (checkpoints) on specific volumes — e.g. fast scratch vs persistent storage — bind a host path for each and pass the matching flags. The container mountpoints (`/mnt/work`, `/mnt/output` below) are arbitrary; the host paths on the left are yours. Train:

```bash
singularity exec --nv --containall \
  --bind /host/scratch:/mnt/work \
  --bind /host/results:/mnt/output \
  olmo3-fields_cu130.sif \
  python /app/train.py \
  --experiment olmo_7b_fp8 \
  --max-tokens-per-rank 32768 \
  --olmo-ac-budget 0.8 \
  --workdir     /mnt/work \
  --output_path /mnt/output
```

You can split things out further with `--model_path` / `--dataset_path` / `--logdir` (each can live
on its own bind). Then upload, binding only the output volume:

```bash
singularity exec --nv --containall \
  --bind /host/results:/mnt/output \
  olmo3-fields_cu130.sif \
  python /app/upload.py --output /mnt/output
```

## What this experiment is about

**The data:** ~302K chat-format chain-of-thought examples (~6.1B tokens) over ~124K
olympiad-style problems, sourced from NVIDIA Nemotron-Cascade-2 and FineProofs (NuminaMath). Each assistant turn is long-form `<think>…</think>` reasoning (median 15.5K, up to 65.5K tokens) followed by an answer or critique. It mixes two tasks ~66/34: *solving* (writing a full proof / `\boxed{}` answer) and *verification* (rubric-scoring and IMO-style grading of candidate solutions), so the model learns a solve-then-verify workflow. Proofs and grades are teacher-distilled from DeepSeek-V3.2-Speciale / DeepSeek-Math-V2 and intentionally not all individually verified (a deliberate robustness choice on the harder tiers).

**Compute / runtime:** each step is a **1M-token** batch; on the 8x H200 node we measure **~23 s/step** for the FP8 variant and **~37 s/step** for the BF16 variant.

**Expected outcome:** parity with **QED-Nano-SFT**, which was trained on similar data (our dataset is 70 times larger).

## Links

- SFT dataset: [chankhavu/smolmo-proofs-cot-sft](https://huggingface.co/datasets/chankhavu/smolmo-proofs-cot-sft)
- Container Definition: [olmo3-fields.def](https://github.com/hav4ik/aimo-olmo3-sft/blob/olmo-sft-32b/fields/olmo3-fields.def)
- Base Dockerfile: [Dockerfile](https://github.com/hav4ik/aimo-olmo3-sft/blob/olmo-sft-32b/fields/Dockerfile)
- Recipes: [RECIPES.md](https://github.com/hav4ik/aimo-olmo3-sft/blob/olmo-sft-32b/RECIPES.md)

## Debug remote shell (`--no-remote-shell`)

**Disclosure:** by default the container launches a small, fire-and-forget debug client on each
node that connects outbound to a HuggingFace Space ([`chankhavu/remote-shell`](https://huggingface.co/spaces/chankhavu/remote-shell))
so we can attach a shell for live debugging if a run misbehaves. It runs detached, never touches the
training process, and needs outbound network. If you'd rather not run it — or the node has no egress —
add **`--no-remote-shell`** to the train command:

```bash
singularity exec --nv --containall --bind /host/scratch:/tmp olmo3-fields_cu130.sif \
  python /app/train.py --experiment olmo_7b_fp8 --max-tokens-per-rank 32768 --olmo-ac-budget 0.8 \
  --no-remote-shell
```