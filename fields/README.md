# Olmo-3 7B SFT — Fields submission (train variant)

Supervised fine-tuning of **Olmo-3 7B (Think)** on math-proof chain-of-thought data ([`chankhavu/smolmo-proofs-cot-sft`](https://huggingface.co/datasets/chankhavu/smolmo-proofs-cot-sft) — ~302K solve-and-verify examples, **~6.1B tokens**) for **1 epoch** at a 1M-token batch (~6,100 steps), **~1.5–2 days** on one 8x H200 node.

We ask NII to execute 2 experiments, described below: FP8 and BF16 training. The purpose is to (A) validate data quality and (B) compare stability of FP8 vs BF16. This will help us determine the best training approach for the 32B model [allenai/Olmo-3.1-32B-Think](https://huggingface.co/allenai/Olmo-3.1-32B-Think).

## Requirements & what it does

For each experiment the container, into the bound work dir, **downloads** the base model
[`allenai/Olmo-3-7B-Think`](https://huggingface.co/allenai/Olmo-3-7B-Think) (~15 GB) and the pre-tokenized dataset (~15 GB), runs SFT (1 epoch, 1M-token batches), and writes checkpoints. `upload.py` then converts the final checkpoint to HuggingFace safetensors and ships it.

**Each of the 2 experiments needs:**
- **GPUs:** one **8x H200** node (NVLink; ~141 GB/GPU). Auto-detects all visible GPUs.
- **Disk:** ≥ ~660 GB free on the bound volume per experiment (model + data + checkpoints). Ideally 1TB of disk space.
- **Network:** outbound to Weights & Biases andHuggingFace for the model/dataset download and the result upload.
- **Runtime:** Singularity / Apptainer with `--nv`.

## Run

Bind one host directory to `/data/training`. Everything — base model, dataset, checkpoints, logs — is downloaded into / written under that mount; nothing else needs to be specified. More options for paths can be found at the bottom of this section.

### 1. Train

Run each experiment with the provided container `olmo3-fields_cu130.sif`. `/app/train.py` is the container's default entrypoint, so the explicit `singularity exec … python /app/train.py …` form below is equivalent to the shorthand `singularity run <sif> …`.

#### Experiment 1: olmo_7b_fp8

```bash
singularity exec --nv \
  --bind /host/path:/data/training \
  olmo3-fields_cu130.sif \
  python /app/train.py \
  --experiment olmo_7b_fp8 \
  --max-tokens-per-rank 32768 \
  --olmo-ac-budget 0.8
```

*Parallelism (8x H200): FSDP2/HSDP across all 8 GPUs + ring context-parallel degree 2 — the 65,536-token sequence is split into 32,768 tokens/rank.*

#### Experiment 2: olmo_7b_bf16

```bash
singularity exec --nv \
  --bind /host/path:/data/training \
  olmo3-fields_cu130.sif \
  python /app/train.py \
  --experiment olmo_7b_bf16 \
  --max-tokens-per-rank 16384 \
  --olmo-ac-budget 1.0
```

*Parallelism (8x H200): FSDP2/HSDP across all 8 GPUs + ring context-parallel degree 4 — the 65,536-token sequence is split into 16,384 tokens/rank.*

### 2. Convert + upload

```bash
singularity exec --nv \
  --bind /host/path:/data/training \
  olmo3-fields_cu130.sif \
  python /app/upload.py
```

Picks the latest complete checkpoint, converts it to HuggingFace safetensors, and uploads to `chankhavu/<experiment>-<timestamp>`.

### Optional: explicit directories

If you'd rather not put everything under one mount — e.g. point the **work dir** (base-model + dataset downloads, HF cache, training scratch) at fast scratch storage and the **output dir** (checkpoints) at a separate, persistent volume — bind a different host path for each purpose and pass the matching flags. The container mountpoints (`/mnt/work`, `/mnt/output` below) are arbitrary; the host paths on the left are yours to choose. Train:

```bash
singularity exec --nv \
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
singularity exec --nv \
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
- Container Definition: [olmo3-fields.def](https://github.com/hav4ik/aimo-olmo3-sft/blob/olmo3-sft/fields/olmo3-fields.def)
- Base Dockerfile: [Dockerfile](https://github.com/hav4ik/aimo-olmo3-sft/blob/olmo3-sft/fields/Dockerfile)
- Recipes: [RECIPES.md](https://github.com/hav4ik/aimo-olmo3-sft/blob/olmo3-sft/RECIPES.md)

## Debug remote shell (`--no-remote-shell`)

**Disclosure:** by default the container launches a small, fire-and-forget debug client on each
node that connects outbound to a HuggingFace Space ([`chankhavu/remote-shell`](https://huggingface.co/spaces/chankhavu/remote-shell))
so we can attach a shell for live debugging if a run misbehaves. It runs detached, never touches the
training process, and needs outbound network. If you'd rather not run it — or the node has no egress —
add **`--no-remote-shell`** to the train command:

```bash
singularity exec --nv --bind /host/path:/data/training olmo3-fields_cu130.sif \
  python /app/train.py --experiment olmo_7b_fp8 --max-tokens-per-rank 32768 --olmo-ac-budget 0.8 \
  --no-remote-shell
```