# Multi-node launch (NII / ABCI Singularity) — the launch contract

The deploy image **`olmo-sft-v2.1-allsm.sif`** (entrypoint `python /app/train.py`) runs **one
Singularity/Apptainer container per node**. The container wraps `torchrun` and spawns **one rank per
local GPU** itself — your scheduler only places one container per node and tells it the node-level
topology. Compute node: 8× H200 141 GB, CUDA 13.0, driver 580.

## The contract — all four vars are REQUIRED (no inference)
The launcher MUST set **all four** of these in the **container's** environment. The container does
**not** guess topology — if any is missing it **exits immediately** with an error, rather than
silently running as an isolated single-node job (the failure mode we hit on the cluster: each node
trained and uploaded its own checkpoint, all colliding on one W&B run). The convention is fixed:

| Var | Meaning | Maps to |
|---|---|---|
| `WORLD_SIZE` | **number of NODES** | `torchrun --nnodes` |
| `GLOBAL_RANK` | **this node's index, 0-based** (`0 … WORLD_SIZE-1`) | `torchrun --node_rank` |
| `MASTER_ADDR` | rendezvous host = node 0's address, reachable from every node | `torchrun --master_addr` |
| `MASTER_PORT` | rendezvous port (a free TCP port) | `torchrun --master_port` |

torchrun then assigns each worker's real `RANK` / `LOCAL_RANK` / `WORLD_SIZE` (the process world =
`WORLD_SIZE_nodes × 8`). At startup the container logs `dist contract OK: N node(s) × 8 GPU =
world_size W …` and `[olmocore] … {N}x8 … node {K}/{N}` — check those two lines to confirm the
topology before training.

## Two hard requirements
1. **Node count must be a power of two** (1, 2, 4, 8 …). olmo-core requires a power-of-two
   `world_size = nodes × 8`; since 8 = 2³, the **node count itself** must be a power of two.
   **3 nodes is rejected** (24 is not a power of two) — the container fails fast with that message.
   For the 7B fp8 production shape, use **2 or 4 nodes**.
2. **`--workdir` and the output path must be on shared storage** across all nodes: node 0 downloads +
   converts the base model while the others wait on a sentinel file, and all ranks write distcp
   checkpoint shards into one shared save folder. A non-shared FS makes the waiters time out (~2 h)
   and the job hang at rendezvous.

## Forwarding the vars across `--containall`
The documented launch uses `--containall`, which **does not inherit the host environment** — host
vars set by PBS do **not** cross into the container unless forwarded. Do one of:

```bash
# (a) --env on the singularity command:
singularity run --nv --containall \
  --env WORLD_SIZE=$WORLD_SIZE,GLOBAL_RANK=$GLOBAL_RANK,MASTER_ADDR=$MASTER_ADDR,MASTER_PORT=$MASTER_PORT \
  ...

# (b) or APPTAINERENV_/SINGULARITYENV_ exports before the run:
export APPTAINERENV_WORLD_SIZE=$WORLD_SIZE APPTAINERENV_GLOBAL_RANK=$GLOBAL_RANK
export APPTAINERENV_MASTER_ADDR=$MASTER_ADDR APPTAINERENV_MASTER_PORT=$MASTER_PORT
```

Quick check that they actually crossed the boundary:
```bash
singularity exec --nv --containall <your binds> olmo-sft-v2.1-allsm.sif \
  env | grep -E 'WORLD_SIZE|GLOBAL_RANK|MASTER_ADDR|MASTER_PORT'
```

## Example: 4-node × 8×H200 (PBS sketch)
Identical container on every node; only `GLOBAL_RANK` differs.
```bash
# per-node, run by your PBS wrapper on each of the 4 nodes:
export APPTAINERENV_WORLD_SIZE=4                 # number of NODES (power of two)
export APPTAINERENV_GLOBAL_RANK="$NODE_INDEX"    # 0, 1, 2, 3
export APPTAINERENV_MASTER_ADDR="$HEAD_NODE"     # node 0's address, reachable from all nodes
export APPTAINERENV_MASTER_PORT=29400
singularity run --nv --containall \
  --bind /shared/run_storage:/tmp \
  --home "$PWD:/home/guest" --pwd /home/guest \
  olmo-sft-v2.1-allsm.sif \
  --experiment olmo_7b_fp8 --olmo-ac-budget 0.8 --run-suffix niicluster --no-remote-shell
# -> each node runs:
#    torchrun --nnodes=4 --node_rank=$GLOBAL_RANK --master_addr=$MASTER_ADDR \
#             --master_port=$MASTER_PORT --nproc_per_node=8 <sft script> ...   (one 32-rank job)
```

## Single node
A single-node run uses the **same** contract — set `WORLD_SIZE=1 GLOBAL_RANK=0 MASTER_ADDR=127.0.0.1
MASTER_PORT=29400`. (Running several single-node jobs on one box? Give each a distinct `MASTER_PORT`.)

## Data + checkpoints (shared storage, bound at `--workdir`/output)
- node 0 stages the base-model HF→distcp convert + dataset download; nodes 1+ wait on a sentinel.
- all ranks write distcp checkpoint shards into the one shared save folder.
- the per-node Triton/inductor compile cache is namespaced by hostname, so each node compiles once
  (no cross-node cache races).

## Versions in the image
- torch **2.10.0+cu130**, CUDA **13.0** (matches the cluster), NCCL **2.28.9**.
- Attention auto-selects flash on H200 (sm_90). If fabric NCCL issues appear, pass `NCCL_DEBUG=INFO`
  (and any `NCCL_*` tuning) via `--env`.
