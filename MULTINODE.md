# Multi-node launch (ABCI / NII Singularity) — interface & example

This is the launch contract for the deploy images on the ABCI cluster (compute node **H**: 8× H200
141 GB, 2 TB RAM, 96 cores; CUDA 13.0, driver 580.105.08). One **Singularity/Apptainer container per
node**; the container wraps `torchrun` and spawns **one rank per local GPU** itself — the scheduler
only deals with node-level placement.

## What the container does for you
- Wraps `torchrun` (olmocore) / `accelerate launch` (axolotl). You do **not** launch torchrun.
- Auto-detects local GPUs → `--nproc_per_node` (override with `NPROC_PER_NODE`). torchrun assigns
  each rank's `RANK` / `LOCAL_RANK` / `WORLD_SIZE`. "Local rank" is handled inside, as requested.
- The nvrtc/TE fix is **baked into the image**; the **code is cloned at startup into
  `/data/training/code`** (the writable run-storage bind), so it runs on a **read-only** squashfs
  with no `--writable-tmpfs` needed. Pin the exact code with `CODE_REF=<branch|tag|commit>` — the
  resolved SHA is logged at startup. Multi-node: node-rank 0 stages the clone, others reuse it.
- For OLMo-core, the one-time HF→distcp checkpoint conversion runs automatically on first use; in a
  multi-node job only node-rank 0 converts and the others wait on a sentinel on the shared storage.

## Environment the container reads
Your PBS script sets these per node (the container maps them to a **static** torchrun rendezvous):

| Var | Meaning | Maps to |
|---|---|---|
| `MASTER_ADDR` | rendezvous host (rank-0 node) | `torchrun --master_addr` |
| `MASTER_PORT` | rendezvous port (default 29400) | `torchrun --master_port` |
| `WORLD_SIZE` | total ranks (standard torchrun) **or** #nodes — auto-detected | `torchrun --nnodes` |
| `GLOBAL_RANK` | this node's base global rank **or** node index — auto-detected | `torchrun --node_rank` |

> **Both conventions are handled automatically.** The container knows its local GPU count, so it
> disambiguates: if `WORLD_SIZE` is a clean multiple of the local GPU count it's treated as
> process-level (standard torchrun: `nnodes = WORLD_SIZE / gpus_per_node`,
> `node_rank = GLOBAL_RANK / gpus_per_node`); otherwise as node-level. Either way you can force it
> with `NNODES` / `NODE_RANK`. The container logs the resolved `node R/N` and the exact `torchrun`
> line at startup, so the topology is visible before any rendezvous.

Run-selection + secrets (via `APPTAINERENV_*` / `SINGULARITYENV_*` or `--env`):

| Var | Meaning |
|---|---|
| `FRAMEWORK` | `olmocore` \| `axolotl` |
| `PRECISION` | `bf16` (default) \| `fp8` |
| `MODEL_SIZE` | axolotl only: `7b` \| `32b` |
| `DATASET_NAME` | prepped dataset under `/data/training/datasets/<NAME>/` |
| `HF_TOKEN` | HuggingFace token (gated base model download) |
| `WANDB_API_KEY` | enables W&B logging (incl. the FP8 `optim/step skipped` metric); unset ⇒ offline |
| `CODE_REF` | pin the code: branch, tag, or commit SHA (default branch `olmo3-sft`) — resolved SHA is logged |

## Build the SIF
```bash
# from a pushed image (recommended), or docker-daemon:// for a local one
apptainer build olmo3-olmocore.sif docker://chankhavu/olmo3-olmocore:cu130
```

## Example: 3-node × 8×H200 run (PBS sketch)
The container is identical on every node; only `GLOBAL_RANK` differs. `MASTER_ADDR/PORT/WORLD_SIZE`
are set by your PBS scaffolding.

```bash
# --- per-node invocation (your PBS wrapper runs this on each of the 3 nodes) ---
apptainer exec --nv \
  --bind /path/to/run_storage:/data/training \
  --env FRAMEWORK=olmocore,PRECISION=bf16,DATASET_NAME=mymath \
  --env HF_TOKEN="$HF_TOKEN",WANDB_API_KEY="$WANDB_API_KEY" \
  olmo3-olmocore.sif
# MASTER_ADDR / MASTER_PORT / WORLD_SIZE / GLOBAL_RANK come from the PBS environment.
# -> container runs: torchrun --nnodes=$WORLD_SIZE --node_rank=$GLOBAL_RANK \
#      --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nproc_per_node=8 <sft script> ...
```
Full recipe defaults (seq 32768 / 1,048,576 tok / 2 epochs) apply when no smoke overrides are set.
For 32B use the axolotl image with `--env MODEL_SIZE=32b` (it maps the same env to
`accelerate launch --num_machines/--machine_rank/--main_process_ip/--main_process_port`).

## Data + checkpoints (shared 1 TB storage, bound to `/data/training`)
- `datasets/<NAME>/` — prepped offline beforehand (see `DATA.md`); read-only at train time.
- `checkpoints/` — written here (the only writable path the run needs besides `/tmp`).
- `hf_cache/`, `wandb/` — created automatically.
All three nodes bind the **same** shared storage so the convert sentinel and checkpoints are visible
cluster-wide.

## Versions in the image
- torch **2.10.0+cu130**, CUDA **13.0** (matches the cluster).
- NCCL **2.28.9** (newer than the recommended 2.23.x; compatible with driver 580 / CUDA 13). If NCCL
  issues appear on the fabric, set `NCCL_*` tuning via `--env` (e.g. `NCCL_DEBUG=INFO`).
- Attention auto-selects `flash_3` (FA3) on H200 (sm_90).
