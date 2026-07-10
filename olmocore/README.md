# Olmo-3 32B SFT (olmo-core) — run knobs reference

## TL;DR — best setting for 128K context (32B, multi-node H100)

128K needs Ulysses CP (activations don't fit otherwise) **and** the model sharded across enough GPUs to
fit the optimizer floor. **cp8 (16384 tokens/rank) is the MAX Ulysses CP for 40 heads at 128K** (cp must
divide both 40 and 131072 → only 1/2/4/8), so you fit by sharding wider, not by more CP. **Minimal
command** (only the knobs that differ from defaults):

```bash
docker run --rm --gpus all --ipc=host -e HF_TOKEN=$HF_TOKEN \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v /host/data:/data/training chankhavu/olmo3-olmocore:cu128-fa2-sink \
  python /usr/local/bin/train.py \
      --seq-len 131072 --max-tokens-per-rank 16384 --cp-style ulysses \
      --ac-budget 0 --nodes-per-fsdp-group 2 --grad-reduce-dtype bf16
```

Per rank ≈ **24 GB floor + ~27 GB activations ≈ ~51 GB** → fits 80 GB H100 (needs ≥2 nodes; a single
8-GPU node can't shard the floor past 8). `--nodes-per-fsdp-group 4` → ~12 GB floor / ~40 GB total for
more headroom (more inter-node comm).

**Every knob for this run, and whether it's already the default** — the command passes only the 6
non-defaults; drop any of those to fall back, or add any of the `(default)` rows to change it:

| Flag | Value | Default? | Note |
|---|---|:--:|---|
| `--seq-len 131072` | 131072 | **no** (65536) | the context window |
| `--max-tokens-per-rank 16384` | 16384 | **no** (auto) | → cp8; max CP for 40 heads @ 128K |
| `--cp-style ulysses` | ulysses | **no** (ring) | **required** for sinks (ring rejects them) |
| `--ac-budget 0` | 0 | **no** (selected_modules) | recompute everything |
| `--nodes-per-fsdp-group 2` | 2 | **no** (1) | shard 32B floor across 16 GPUs |
| `--grad-reduce-dtype bf16` | bf16 | **no** (fp32) | ~8 GB/rank less |
| `--optim skip_step` | skip_step | ✅ default | SkipStepAdamW spike protection |
| `--optim-dtype bf16` | bf16 | ✅ default | bf16 moments, ~16 GB/rank less |
| `--fused-lce 1` | 1 | ✅ default | no materialized logits, ~10 GB+ |
| `--fused-rmsnorm` (omit) | auto | ✅ default | auto-off on H100 (big smem) |
| `--attn-backend` (omit) | auto | ✅ default | flash_3 on H100 by arch |
| `--sink 1` | 1 | ✅ default | per-head attention sink |
| `--epochs 2` | 2 | ✅ default | `--max-steps` overrides |
| `--gbs 1572864` | 1.5M | ✅ default | tokens/optimizer step |
| `--keep-ckpts 3` | 3 | ✅ default | ~1 TB distcp on disk |
| `--save-interval 1000` / `--ephemeral-interval 500` | 1000/500 | ✅ default | checkpoint cadence |
| `--lr 5e-5` | 5e-5 | ✅ default | |
| `--model-dtype` (omit) | float32 | ✅ default | bf16 master = risky, no SR |
| `--code-ref olmocore-cu128-fa2-sink` | — | ✅ default | branch cloned at runtime |

Full details for every knob below.

---

Every tunable for the olmo-core SFT recipe, in one place: optimizer, parallelism, sequence/batch
sizing, memory, activation checkpointing, fused ops, attention sink, and checkpointing.

The container image is `chankhavu/olmo3-olmocore:cu128-fa2-sink`. The **recipe** (this repo, the SFT
script + `run.sh`) is **cloned at runtime** from the branch you pass as `--code-ref`, so recipe changes
take effect on the next run **without an image rebuild**. Only the baked `train.py` CLI itself needs a
rebuild to gain new `--flags` — but **every flag has an `OLMO_*` env equivalent that works today**.

---

## Quick start

```bash
docker pull chankhavu/olmo3-olmocore:cu128-fa2-sink
docker run --rm --gpus all --ipc=host -e HF_TOKEN=$HF_TOKEN \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v /host/data:/data/training chankhavu/olmo3-olmocore:cu128-fa2-sink \
  python /usr/local/bin/train.py \
      --seq-len 65536 --max-tokens-per-rank 8192 --cp-style ulysses \
      --ac-budget 0 --optim skip_step --optim-dtype bf16 \
      --code-ref olmocore-cu128-fa2-sink
```

`python /usr/local/bin/train.py --help` lists everything; `--dry-run` prints the resolved env without
launching. Anything you don't pass falls back to `run.sh`'s per-model defaults.

### How a run is wired
```
train.py (--flags -> OLMO_* env)  ->  bootstrap.sh (git-clone --code-ref)  ->  run.sh (per-size defaults)
  ->  torchrun  ->  Olmo-3-32B-SFT-bf16.py (reads OLMO_* env)
```
Flag ⇄ env: `--optim skip_step` is identical to `-e OLMO_OPTIM=skip_step`. Use the env form to enable a
flag the baked `train.py` doesn't have yet.

---

## Sequence length & batch (per-device, per-DP-group, global)

| Knob | Env | What it controls |
|---|---|---|
| `--seq-len` | `SEQ_LEN` | length of ONE training sequence = the context window (65536, 131072, …) |
| `--max-tokens-per-rank` | `OLMO_MAX_TOKENS_PER_RANK` | per-GPU token cap → sets **cp_degree** = ⌈seq_len / cap⌉ |
| `--gbs` | `GLOBAL_BATCH_SIZE` | **global batch = tokens per optimizer step** (all DP replicas). Default 1,572,864 |
| `--epochs` / `--max-steps` | `EPOCHS` / `MAX_STEPS` | duration (max-steps overrides epochs) |

**The three "sequence sizes" people conflate:**
- **Global / full context** = `--seq-len` (e.g. 131072). One document/packed sample is this long.
- **Per-device (per-rank) tokens** = `seq_len / cp_degree`. Context parallelism (Ulysses) splits *one
  sequence* across `cp_degree` GPUs. You set this indirectly: `--max-tokens-per-rank 8192` → cp8 →
  8192 tok/rank for a 65536 seq (or 16384 tok/rank for 131072).
- **Per-DP-group tokens/step** = `gbs / dp_replicas`, processed as `grad_accum` microbatches of
  `rank_microbatch` each. Grad-accum = `gbs / (rank_microbatch × dp_world)`.

**Activation memory scales with per-device tokens**, not seq-len — that's why CP is how you fit long
context. Constraint: `cp_degree` must divide `n_heads` (40 → cp ∈ {1,2,4,5,8,…}) for Ulysses.

---

## Parallelism

| Knob | Env | Notes |
|---|---|---|
| `--cp-style` | `OLMO_CP_STYLE` | `ulysses` (all-to-all; **required with attention sink**) or `ring`. Ring **rejects sinks** (incremental LSE can't apply the per-head correction). cp=1 (no CP) is also fine for sinks. |
| `--max-tokens-per-rank` | `OLMO_MAX_TOKENS_PER_RANK` | sets cp_degree (see above) |
| `--nodes-per-fsdp-group` | `OLMO_NODES_PER_FSDP_GROUP` | **how wide the model shards.** 1 (default) = shard within a node, replicate across nodes. Raise to shard the model+optimizer floor across more nodes. |
| `--nnodes` / `--node-rank` / `--master-addr` / `--master-port` | `WORLD_SIZE` / `GLOBAL_RANK` / `MASTER_ADDR` / `MASTER_PORT` | multi-node rendezvous |

**Critical mental model — CP vs sharding:**
- **CP splits the *sequence*** → cuts *activation* memory. It does **not** shrink the model/optimizer floor.
- **FSDP shard width splits the *model*** → cuts the *floor*. One copy of the sharded model lives across
  `shard_degree × cp_degree` GPUs (CP ranks join the shard group via olmo-core's `dp_cp` flatten).
- Default HSDP shards **intra-node only** (`shard_degree = GPUS_PER_NODE / cp`), so **adding nodes only
  adds replicas (more batch), not less memory** — until you raise `--nodes-per-fsdp-group`.

| `--nodes-per-fsdp-group` (cp8, 8-GPU nodes) | shards over | floor (32B) |
|---|---|---|
| 1 (default) | 8 | ~40 GB |
| 2 | 16 | ~24 GB |
| 4 | 32 | ~12 GB |
| = #nodes (all) | all | ~5 GB |

Wider shard = lower floor, but the FSDP all-gather crosses more nodes (inter-node IB vs NVLink). The
sweet spot is a group of 2–4 nodes: fits the model, keeps most heavy comm local.

---

## Optimizer

| Knob | Env | Notes |
|---|---|---|
| `--optim` | `OLMO_OPTIM` | **`skip_step` (DEFAULT)** — SkipStepAdamW spike protection (equals AdamW when not skipping) · `fused_adamw` (fp32 fused-kernel baseline, fastest, most VRAM) |
| `--optim-dtype` | `OLMO_OPTIM_DTYPE` | **DEFAULT `bf16`** for skip_step — Adam moments (m/v) in bf16 (**~16 GB/rank less**, fp32 master kept). Set `fp32` to force fp32 moments. Ignored by `fused_adamw`. |

**Default = `skip_step` + bf16 moments** — spike protection + the biggest safe optimizer memory cut, the
proven config from the fp8 run. Nothing to pass; use `--optim fused_adamw` only for the fp32 baseline.

> ⚠️ **`adamw8bit` (bitsandbytes) does NOT work here.** bitsandbytes has no DTensor support, so under
> FSDP2 its 8-bit update kernel raises *"optimizer_update_8bit_blockwise got mixed torch.Tensor and
> DTensor"*. It fails on H100 too (DTensor issue, not kernel-arch). The recipe now rejects it at config
> time. There is **no working 8-bit optimizer** for this FSDP2 setup.

Optimizer state per rank (sharded over `shard_degree × cp`): fp32 master (4 B/param) + moments
(fp32 = 8 · or bf16 = 4 B/param) + gradients (see below).

---

## Memory-saving knobs (the whole ladder)

Per-rank memory = **floor** (model + optimizer, sharded) + **activations** (per-device tokens) +
overhead. Floor ≈ `(4 master + moments + grad_bytes) · P / (shard_degree · cp)`.

| Knob | Env | Saves | Risk |
|---|---|---|---|
| `--nodes-per-fsdp-group N` | `OLMO_NODES_PER_FSDP_GROUP` | floor × (1/N) — **the big lever** | none (more inter-node comm) |
| `--optim skip_step --optim-dtype bf16` | `OLMO_OPTIM` / `OLMO_OPTIM_DTYPE` | ~16 GB/rank (bf16 moments) | validate loss w/ grad-accum |
| `--grad-reduce-dtype bf16` | `OLMO_GRAD_REDUCE_DTYPE` | ~8 GB/rank (bf16 grad reduce-scatter) | low — fp32 master absorbs it; watch grad-accum |
| fused-LCE (**default on**) | `OLMO_FUSED_LCE` | ~10 GB+ (no materialized (T,vocab) logits) | none (z-loss×skip-step gap fixed upstream) |
| `--max-tokens-per-rank ↓` (more CP) | `OLMO_MAX_TOKENS_PER_RANK` | activations × (cp/cp') | none |
| `--ac-budget 0` | `OLMO_AC_BUDGET` | activations (recompute all) | slower |
| `--model-dtype bfloat16` (bf16 master) | `OLMO_MODEL_DTYPE` | ~8 GB/rank | **med — no stochastic rounding in olmo-core; validate loss** |

**Not available** (would need fork patches, not exposed by olmo-core): FSDP `CPUOffloadPolicy`
(optimizer/params → host RAM, ~40 GB off GPU, ~2–3× slower), activation CPU offload (~10 GB), liger
SwiGLU MLP (low ROI under recompute). olmo-core wires exactly **one** liger kernel — fused-linear-CE,
already on. No liger SwiGLU/RoPE/RMSNorm.

### Rough floor + budget for 32B (P = 32.2 B)
| Component (skip_step + bf16 moments + bf16 grad, shard-8) | GiB |
|---|---|
| fp32 master (4·P/8) | 16 |
| bf16 grad (2·P/8) | 8 |
| bf16 Adam m + v (2+2·P/8) | 16 |
| **Floor** | **~40** |
| Activations @ 16384 tok/rank (ac 0) | ~15–27 |
| **Total (8-way shard)** | **~60–70** |

To go below ~40 GB floor on 8 GPUs you **must** shard wider (`--nodes-per-fsdp-group`) or CPU-offload —
those are the only additive levers left.

---

## Activation checkpointing

| `--ac-budget` (`OLMO_AC_BUDGET`) | Mode | Memory / speed |
|---|---|---|
| unset | `selected_modules` (recompute every FFN) | AI2 reference; moderate |
| `0..1` | `budget` (torch.compile partitioner; 1=save all, 0=recompute all) | **0 = least memory**, slower; needs compile |
| `none` | disabled | most memory, fastest (only for big cards) |

Long-context, memory-bound → `--ac-budget 0`. Roomy card → `~0.8` (recompute less = faster). AI2:
7B=0.7, 32B=0.3, hybrid-SFT=0.1.

---

## torch.compile / fused ops

| Knob | Env | Notes |
|---|---|---|
| `--fused-rmsnorm` | `OLMO_FUSED_RMSNORM` | `1`/`0`; **auto-on** when GPU opt-in smem < 200 KB (RTX 6000/A100). Routes the wide (5120) RMSNorms — block, lm_head **and q_norm** (5120-wide since `use_head_qk_norm=False`) — through flash-attn's Triton kernel. Fixes the *"out of resource: shared memory / No valid triton configs"* compile OOM. Hopper/B200 keep stock `rms`. |
| `--persistent-reductions` | `OLMO_PERSISTENT_REDUCTIONS` | `1`/`0`; auto-off on small-smem GPUs. Defence-in-depth; the real fix for the wide-norm OOM is `fused_rmsnorm`. |
| fused-LCE | `OLMO_FUSED_LCE` | default **1** (on); `0` = materialized reference for an A/B. |

Eager fallback (no compile at all): `-e TORCHDYNAMO_DISABLE=1` (drop `--ac-budget`; budget mode needs
compile). Compile of all 64 layers happens once in the dry-run (~15 min); steps after are fast.

---

## Attention sink

| Knob | Env | Notes |
|---|---|---|
| `--sink` | `OLMO_USE_SINK` | `1` (default) adds a per-head learnable sink logit per layer |
| `--sink-init` | `OLMO_SINK_INIT` | initial logit for a **stock** (no-sink) warm start; leave unset when the checkpoint already carries trained sinks |
| `--self-check` | `OLMO_ATTN_SELFCHECK` | `1` runs the sink-kernel self-check before training |

Sinks use exact post-correction on FA2/FA3 (`o/(1+exp(sink−lse))`). They **require the complete softmax
LSE per query**, so: cp=1 ✅, **Ulysses** ✅ (full seq per head-slice), **ring** ❌ (rejected).

---

## Checkpointing

| Knob | Env | Default | Notes |
|---|---|---|---|
| `--keep-ckpts` | `OLMO_KEEP_LAST_CKPTS` | **3** (32B) | max PERSISTENT checkpoints on disk; oldest deleted. `0` = keep all |
| `--save-interval` | `OLMO_SAVE_INTERVAL` | 1000 | steps between persistent checkpoints |
| `--ephemeral-interval` | `OLMO_EPHEMERAL_INTERVAL` | 500 | steps between rotating resume checkpoints (must be < save-interval) |
| `--stage` | `STAGE` | train | `convert` = HF→distcp only |

On disk at once ≈ `keep_last` persistent + 1 ephemeral. 32B distcp ≈ **251 GB each**, so **keep-3 ≈
~1 TB** — confirm the FS has room, or drop to `--keep-ckpts 1` on a tight disk. Keeping the last few
also feeds checkpoint-soup / TIES merging.

---

## Model & data

| Knob | Env | Default |
|---|---|---|
| `--model` | `HF_MODEL` | `chankhavu/yccchen-olmo3-deploy` |
| `--dataset` / `--dataset-subdir` / `--dataset-name` | `DATASET_HF` / `DATASET_SUBDIR` / `DATASET_NAME` | tokenized dataset repo |
| `--model-size` | `MODEL_SIZE` | `32b` (picks `run.sh` defaults) |
| `--sft-script` | `SFT_SCRIPT_NAME` | `Olmo-3-32B-SFT-bf16.py` (FP8-free) |
| `--hf-tokenizer` | `OLMO_HF_TOKENIZER` | `1` (reuse model tokenizer) or an HF id |
| `--lr` | `LR` | 5e-5 |
| `--code-ref` | `CODE_REF` | `olmocore-cu128-fa2-sink` (branch cloned at runtime) |
| `--data` | `DATA` | `/data/training` |

---

## Recommended configs

**Single 8-GPU node, 64K (test / borderline on 94 GB):**
```bash
--seq-len 65536 --max-tokens-per-rank 8192 --cp-style ulysses --ac-budget 0 \
--optim skip_step --optim-dtype bf16     # + OLMO_GRAD_REDUCE_DTYPE=bf16
```

**64× H100 (80 GB), 128K — shard wider, no offload needed:**
```bash
OLMO_NODES_PER_FSDP_GROUP=2 OLMO_GRAD_REDUCE_DTYPE=bf16 \
python /usr/local/bin/train.py --seq-len 131072 --max-tokens-per-rank 16384 \
    --cp-style ulysses --ac-budget 0
```
(`--max-tokens-per-rank 16384` → cp8 = 16384 tok/rank, the max Ulysses CP for 40 heads at 128K;
`8192` would be cp16, invalid since 16 ∤ 40. shard-over-16 → ~24 GB floor → ~51 GB/rank; skip_step+bf16
optimizer is the default.)

**65K without CP** (teammate's H200 shape, ported): shard over all GPUs, `--cp-style` cp=1
(`--max-tokens-per-rank ≥ seq_len`), `OLMO_NODES_PER_FSDP_GROUP=<#nodes>`. Simpler — no Ulysses, no
sink-CP interaction. 128K needs CP regardless (no-CP activations exceed 80 GB).

---

## Running on AI2 Beaker

The image runs on Beaker as a custom Docker image. **Omit `command:`** in the spec so Beaker runs the
image's ENTRYPOINT (`bootstrap.sh`) as-is — it clones the recipe and downloads model+data at runtime
(Beaker jobs have outbound network). `bootstrap.sh` auto-maps Beaker's rendezvous env, so **you set
nothing extra for multi-node**:

| Beaker injects | mapped to |
|---|---|
| `BEAKER_REPLICA_COUNT` | `WORLD_SIZE` (#nodes) |
| `BEAKER_REPLICA_RANK` | `GLOBAL_RANK` (node index) |
| `BEAKER_LEADER_REPLICA_HOSTNAME` | `MASTER_ADDR` |

Single-node = `resources.gpuCount: 8`. Multi-node = `replicas: N` + **`leaderSelection: true`** +
`hostNetworking: true` + `propagateFailure/Preemption: true` (leaderSelection is required or the leader
hostname is unset and torchrun can't rendezvous — `bootstrap.sh` warns if so). Tuning knobs go in
`envVars` as their `OLMO_*` form (e.g. `{name: OLMO_NODES_PER_FSDP_GROUP, value: "4"}`).

- **Storage:** mount a WEKA bucket **read-write** at `/data/training` (`datasets: [{mountPath:
  /data/training, source: {weka: <bucket>}}]`) — that's where checkpoints land and persist. Don't use
  result-datasets for the ~251 GB distcp checkpoints; `result.path` is for small logs only.
- **Secrets:** `beaker secret write HF_TOKEN …`, then `envVars: [{name: HF_TOKEN, secret: HF_TOKEN}]`.
- **Cluster:** target H100 (sm_90) — the image supports it. Multi-node throughput needs InfiniBand
  user-space drivers in the image (TODO; else NCCL falls back to Ethernet ~6× slower).
