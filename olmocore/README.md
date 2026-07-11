# Olmo-3 32B SFT (olmo-core) — run knobs reference

## TL;DR — best setting for 128K context (32B) on AI2 Beaker

128K needs Ulysses CP (activations don't fit otherwise) **and** the model sharded across enough GPUs to
fit the optimizer floor. **cp8 (16384 tokens/rank) is the MAX Ulysses CP for 40 heads at 128K** (cp must
divide both 40 and 131072 → only 1/2/4/8), so you fit by sharding wider, not by more CP.

Submit this Beaker spec (`beaker experiment create spec.yaml`) — 8 nodes × 8 H100 = 64 GPUs, model
sharded across 4-node FSDP groups:

```yaml
version: v2
budget: ai2/<your-budget>
description: Olmo-3 32B SFT @ 128K
tasks:
- name: sft
  image:
    docker: chankhavu/olmo3-olmocore:cu128-fa2-sink
  command:                                    # train.py --flags; it execs the entrypoint (rendezvous shim runs)
  - python
  - /usr/local/bin/train.py
  - --seq-len=131072
  - --max-tokens-per-rank=16384               # cp8 = 16384 tok/rank (max Ulysses CP for 40 heads @ 128K)
  - --cp-style=ulysses
  - --ac-budget=0                            # recompute all = least activation memory (raise to 0.3 if you have headroom)
  - --nodes-per-fsdp-group=4                  # shard the 32B floor across 4 nodes (32 GPUs)
  - --grad-reduce-dtype=bf16
  - --gbs=4194304
  - --epochs=2
  # --- identity: all DEFAULTS below (shown so you know where to change them) ---
  - --model=chankhavu/yccchen-olmo3-deploy               # BASE model to SFT from (Olmo-3 + per-head sinks + Deepseek tokenizer)
  - --dataset=chankhavu/yccchen-stage2-olmocore-256k-v2  # tokenized SFT dataset (HF repo)
  # checkpoint upload sub-path: DEFAULT is ycchen-olmo32b-ds-sft-<timestamp> (run-unique, so re-runs don't
  # overwrite). Leave it out to get the timestamp; uncomment to PIN a fixed name (used verbatim, no timestamp):
  # - --hf-upload-prefix=my-run-name
  replicas: 8                                 # 8 nodes; × gpuCount 8 = 64 GPUs
  leaderSelection: true                       # REQUIRED — the rendezvous shim needs the leader hostname
  hostNetworking: true                        # REQUIRED for InfiniBand (RDMA); WEKA mounts don't need it
  propagateFailure: true
  propagatePreemption: true
  synchronizedStartTimeout: 15m
  resources:
    gpuCount: 8                               # per node
  constraints:
    cluster: [ ai2/jupiter-cirrascale-2 ]     # H100 = sm_90
  datasets:
  - mountPath: /data/training
    source: { weka: <your-weka-bucket> }      # ~1 TB read-write; checkpoints persist here
  envVars:
  - { name: HF_TOKEN,           secret: HF_TOKEN }
  - { name: WANDB_API_KEY,      secret: WANDB_API_KEY }
  - { name: PYTORCH_ALLOC_CONF, value: "expandable_segments:True" }
  - { name: NCCL_SOCKET_IFNAME, value: ib }              # InfiniBand (jupiter)
  - { name: NCCL_IB_HCA,        value: "^=mlx5_bond_0" }  # InfiniBand HCA (jupiter)
  - { name: OLMO_HF_UPLOAD_REPO, value: <hf-user>/olmo3-32b-sft-128k }  # auto-ship each ckpt to HF (HF_TOKEN needs WRITE scope)
  # (no result.path: nothing writes /results — logs + checkpoints live on the WEKA mount below)
  timeout: 48h
```

- **Rendezvous is automatic** — `bootstrap.sh` maps `BEAKER_REPLICA_COUNT/RANK/LEADER_REPLICA_HOSTNAME` →
  `WORLD_SIZE`/`GLOBAL_RANK`/`MASTER_ADDR` (needs `leaderSelection: true`). Nothing to set by hand.
- **Memory** — shard across 4 nodes (32 GPUs) → ~12 GB optimizer floor; `--ac-budget 0` recomputes all
  (least activation memory). At 128K the step is comm-bound so ac 0's throughput cost is modest. The two
  memory knobs trade off: `--ac-budget 0` frees enough activations to *narrow* `--nodes-per-fsdp-group`
  (e.g. → 2, ~24 GB floor) which cuts inter-node comm — so ac 0 + a narrower group can be **both smaller
  and faster** than ac 0.3 + a wider group. Raise ac toward 0.3 only if you have headroom to spare.
- **InfiniBand** — `hostNetworking: true` exposes the fabric; the two `NCCL_*` vars point at jupiter's HCA
  (`mlx5_bond_0`). The image already ships the user-space RDMA libs. Verify with `NCCL_DEBUG=INFO` →
  expect `NET/IB` (not `NET/Socket`).
- **Storage** — WEKA read-write at `/data/training`; checkpoints (~251 GB each, keep-3 ≈ 1 TB) persist
  there and are retrievable from a follow-up job. Don't route them through a `result` dataset.
- **Secrets** — `beaker secret write HF_TOKEN <val>` (+ `WANDB_API_KEY`) in the same workspace first.

skip_step + bf16 moments and fused-LCE are the defaults — no need to pass them.

**Every knob for this run, and whether it's already the default** — the command passes the non-default
knobs (plus `--epochs`/`--gbs`, called out because they matter); drop any to fall back, or add any
`(default)` row to change it:

| Flag | Value | Default? | Note |
|---|---|:--:|---|
| `--seq-len 131072` | 131072 | **no** (65536) | the context window |
| `--max-tokens-per-rank 16384` | 16384 | **no** (auto) | → cp8; max CP for 40 heads @ 128K |
| `--cp-style ulysses` | ulysses | **no** (ring) | **required** for sinks (ring rejects them) |
| `--ac-budget 0` | 0 | **no** (selected_modules) | recompute all = least activation memory; comm-bound at 128K so modest speed cost (0.3 = a bit faster if you have headroom) |
| `--nodes-per-fsdp-group 4` | 4 | **no** (1) | shard 32B floor across 4 nodes / 32 GPUs |
| `--grad-reduce-dtype bf16` | bf16 | **no** (fp32) | ~8 GB/rank less |
| `--optim skip_step` | skip_step | ✅ default | SkipStepAdamW spike protection |
| `--optim-dtype bf16` | bf16 | ✅ default | bf16 moments, ~16 GB/rank less |
| `--fused-lce 1` | 1 | ✅ default | no materialized logits, ~10 GB+ |
| `--fused-rmsnorm` (omit) | auto | ✅ default | auto-off on H100 (big smem) |
| `--attn-backend` (omit) | auto | ✅ default | flash_3 on H100 by arch |
| `--sink 1` | 1 | ✅ default | per-head attention sink |
| `--epochs 2` | 2 | ✅ default | shown because it matters; `--max-steps` overrides |
| `--gbs 4194304` | 4.19M | **no** (1.57M) | tokens/optimizer step; matches AI2's reference batch |
| `--keep-ckpts 3` | 3 | ✅ default | ~1 TB distcp on disk |
| `--save-interval 1000` / `--ephemeral-interval 500` | 1000/500 | ✅ default | checkpoint cadence |
| `--lr 5e-5` | 5e-5 | ✅ default | |
| `--model-dtype` (omit) | float32 | ✅ default | bf16 master = risky, no SR |
| `--model chankhavu/yccchen-olmo3-deploy` | deploy | ✅ default | **base model to SFT from** (sink + Deepseek); change to fine-tune a different base |
| `--dataset chankhavu/yccchen-stage2-olmocore-256k-v2` | yccchen-256k | ✅ default | **tokenized SFT dataset** (HF repo); `--dataset-subdir` if shards are in a subfolder |
| `--hf-upload-repo <user>/…` | — | **no** (unset) | destination repo for the checkpoint watchdog; **required to upload** (HF_TOKEN WRITE scope) |
| `--hf-upload-prefix` (omit) | `ycchen-olmo32b-ds-sft-<ts>` | ✅ default | **checkpoint upload sub-path** in the repo; default is run-unique (timestamped); pin a fixed name to override |
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
context. Constraint: Ulysses scatters the head axis for q AND k/v, so `cp_degree` must divide
`gcd(n_heads, n_kv_heads) = gcd(40, 8) = 8` → **cp ∈ {1,2,4,8}** (the `n_kv_heads=8` GQA bound is tighter
than 40; e.g. cp=5 is invalid).

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

Wider shard = lower floor, but the FSDP all-gather crosses more nodes (inter-node IB vs NVLink). Note at
**cp8 the shard all-gather is entirely inter-node** anyway — cp is the fastest-varying mesh dim, so all 8
GPUs of a node form one CP group and the `dp_shard` dim strides across nodes (there's no intra-node
sharding left to keep local). So pick the group width by how much floor you must shed vs how much
inter-node all-gather you can afford; 2–4 nodes/group is the usual balance (fits the model without going
full-mesh), not a comm-locality win.

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

### How many nodes to fit 128K vs 256K (H100 80 GB)

Two independent drivers, and they don't substitute:
- **Floor** (model+optimizer) shrinks with nodes: `floor/rank ≈ 322 GB ÷ (nodes×8)`.
- **Activations** shrink with CP — but **CP is capped at 8** (the 8 KV heads), so per-rank tokens = `seq_len÷8`
  regardless of node count. **More nodes cut only the floor, never the activations.**

| nodes (GPUs) | floor/rank | **128K** total (act ~27 GB) | **256K** total (act ~42 GB) |
|---|---|---|---|
| 1 (8)  | 40 GB | ~67 GB ⚠️ borderline | ~82 GB ❌ |
| **2 (16)** | 20 GB | **~47 GB ✅** | **~62 GB ✅** |
| 4 (32) | 10 GB | ~37 GB ✅ | ~52 GB ✅ roomy |
| 8 (64) | 5 GB  | ~32 GB | ~47 GB |

*(The ~27 GB @128K activation is validated against a real 94 GiB OOM: that run used adamw8bit = fp32
moments ≈ 64 GB floor, and 94 − 64 ≈ 30 GB of activations.)*

- **128K → 2 nodes** (16 GPUs). 1 node is borderline (~67 GB nominal, ±10 GB of NCCL/frag can tip it over).
- **256K → 2 nodes minimum** (~62 GB), **4 for comfort**. 1 node cannot: activations alone (~42 GB) + a
  40 GB floor exceeds 80 GB.

```bash
# 128K on 2 nodes
--seq-len 131072 --max-tokens-per-rank 16384 --nodes-per-fsdp-group 2 --ac-budget 0
# 256K on 2 nodes (or --nodes-per-fsdp-group 4 for margin)
--seq-len 262144 --max-tokens-per-rank 32768 --nodes-per-fsdp-group 2 --ac-budget 0
```

Because CP maxes at 8, activation cost is **fixed per seq-len** — beyond the minimum that fits, more nodes
only add headroom/throughput, not longer context. Only more KV heads (higher CP cap) or activation offload
would unlock past that.

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

### Auto-upload to HuggingFace (`--hf-upload-repo` / `OLMO_HF_UPLOAD_REPO`)
Set a repo id and a **node-0 background watchdog** (`olmocore/upload.py`, ported from the Fields FP8
pipeline) polls the checkpoint dir every `OLMO_HF_UPLOAD_INTERVAL` s (default 300); for each **new
complete** distcp checkpoint it converts distcp→HF (**sink-preserving** — via the sink-aware
`save_hf_model`), shards the safetensors, and uploads to `<repo>/step<N>/`, writing an
`upload_successful.txt` marker so each ships once. **Three upload triggers** (matching the NII run): (1)
each **new checkpoint** mid-run (the watcher), (2) a **crash** — the final upload runs on a non-zero exit
and ships the **latest survivable** complete checkpoint, and (3) **normal end** — the final upload lands
the end-of-run model. The final upload goes to the **repo root** (so `AutoModel.from_pretrained("<repo>")`
gives it; intermediates live under `step<N>/`). This is how you pull the model **out of the cluster** —
otherwise checkpoints only sit on WEKA/local disk.

- Needs **`HF_TOKEN` with WRITE scope**. Repo is **private** by default (`OLMO_HF_UPLOAD_PRIVATE=0` for public).
- **`--hf-upload-prefix` / `OLMO_HF_UPLOAD_PREFIX`**: a path prefix inside the repo — checkpoints land at
  `<repo>/<prefix>/step<N>` and the final at `<repo>/<prefix>`. **Default is run-unique:
  `ycchen-olmo32b-ds-sft-<YYYYMMDDHHMMSS>`** (a timestamp so re-runs don't overwrite each other, like the
  FP8/NII run — computed once per run, shared by the watcher + final). Override with a fixed name
  (`--hf-upload-prefix 128k-run1`, used verbatim) or pass `""` for the repo root.
- The exported tokenizer follows the trained model — the deploy model's **deepseek** tokenizer (never dolma2).
- CPU-only (`CUDA_VISIBLE_DEVICES=""`) so it never steals a training GPU; convert+ship bounded by
  `OLMO_HF_UPLOAD_TIMEOUT` (default 5400 s) so a wedged upload can't stall the loop; scoped to the current
  `RUN_NAME` so a prior run's checkpoint on a shared volume isn't shipped by mistake.
- Runtime-cloned (`run.sh` + `upload.py`) — **works with the current image, no rebuild** (the
  `--hf-upload-prefix` *flag* form bakes on the next build; the `OLMO_HF_UPLOAD_PREFIX` env works today).

**Smoke-test the upload in a short run.** Even **`--max-steps 10` uploads a model**: olmo-core's
checkpointer saves a **final permanent checkpoint at end-of-run** (`post_train`, whenever the last step is
past the previous save), so the **final upload ships it** to `<repo>/<prefix>` (repo root) — you do *not*
need to reach the 1000-step `save-interval`. The per-checkpoint *watcher* stays quiet in a 10-step run (no
mid-run checkpoint), so the FINAL upload is what you'll see. Add the upload knobs to the run:

```yaml
  command:
  - python
  - /usr/local/bin/train.py
  - --max-steps=10
  - --seq-len=65536
  - --max-tokens-per-rank=16384
  - --cp-style=ulysses
  - --ac-budget=0
  - --hf-upload-repo=<hf-user>/olmo3-smoke     # HF_TOKEN secret needs WRITE scope
```
The final convert+ship of the 32B runs **after** step 10 (CPU distcp→HF convert + ~64 GB upload = a few
minutes) — watch `hf_upload_watcher.log` on the WEKA mount + the tail of run-rank0. To also exercise the
*watcher* (intermediate uploads to `step<N>/`), add `--save-interval=5 --hf-upload-interval=60` so a
mid-run checkpoint is saved and polled while training is still going.

---

## Model & data

| Knob | Env | Default |
|---|---|---|
| `--model` | `HF_MODEL` | `chankhavu/yccchen-olmo3-deploy` |
| `--dataset` / `--dataset-subdir` / `--dataset-name` | `DATASET_HF` / `DATASET_SUBDIR` / `DATASET_NAME` | tokenized dataset repo |
| `--model-size` | `MODEL_SIZE` | `32b` (picks `run.sh` defaults) |
| `--sft-script` | `SFT_SCRIPT_NAME` | `Olmo-3-32B-SFT-bf16.py` (FP8-free) |
| `--hf-tokenizer` | `OLMO_HF_TOKENIZER` | `1` (reuse model tokenizer) or an HF id |
| `--lr` | `LR` | 5e-5 — **peak** LR |
| `--lr-alpha-f` | `OLMO_LR_ALPHA_F` | 0.1 — **floor** = alpha_f × peak (→ 5e-6); 0 = decay to 0 |
| `--warmup-fraction` | `OLMO_LR_WARMUP` | 0.03 — LR warmup as a fraction of total steps |
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

**64× H100 (80 GB), 256K — same recipe, 2× the per-rank tokens:**
```bash
OLMO_NODES_PER_FSDP_GROUP=2 OLMO_GRAD_REDUCE_DTYPE=bf16 \
python /usr/local/bin/train.py --seq-len 262144 --max-tokens-per-rank 32768 \
    --cp-style ulysses --ac-budget 0
```
(`--max-tokens-per-rank 32768` → still cp8 = 32768 tok/rank, since CP is capped at 8 by the KV heads — so
256K just doubles the per-rank tokens vs 128K, ~42 GB of activations. That fixed activation cost means
1 node can't fit it (40 GB floor + 42 GB > 80); **2 nodes is the minimum** (~20 GB floor → ~62 GB/rank),
`--nodes-per-fsdp-group 4` for more headroom (~10 GB floor → ~52 GB). Prefer the narrower group (2) if it
fits — less inter-node all-gather = faster. See the node-sizing table above.)

**65K without CP** (teammate's H200 shape, ported): shard over all GPUs, `--cp-style` cp=1
(`--max-tokens-per-rank ≥ seq_len`), `OLMO_NODES_PER_FSDP_GROUP=<#nodes>`. Simpler — no Ulysses, no
sink-CP interaction. 128K needs CP regardless (no-CP activations exceed 80 GB).

---

## Unit tests (sink / FA2 / FA3 / Ulysses / CP)

One script runs every attention-sink, FA2/FA3, Ulysses-CP, ring, RoPE and RMSNorm test — olmo_core's own
suite (baked in the image) plus this recipe's fused-RMSNorm tests:

```bash
docker run --rm --gpus all --entrypoint bash \
    chankhavu/olmo3-olmocore:cu128-fa2-sink \
    /data/training/code/olmocore/run_tests.sh
```

- Needs **`--gpus all`** — the sink/FA2/FA3 numeric tests build real kernels; without a GPU they error, not skip.
- **1 GPU** exercises the sink/flash/RoPE/norm tests; the **Ulysses / ring / context-parallel** suites self-skip
  unless **≥2 GPUs** are visible (they spawn a small distributed group) — run on an 8-GPU box to cover them.
- `--self-check` also runs the standalone kernel consistency check (eager vs FA2/FA3, FA2 vs FA3) — the same
  one `OLMO_ATTN_SELFCHECK=1` runs as a pre-training pre-flight.
- `--quick` skips the distributed suites; extra args (`-k sink`, `-x`, `-v`) forward to pytest.

The script lives in this repo (cloned to `/data/training/code` at runtime), so it tracks the branch you run —
no rebuild to update tests.

---

## Running on AI2 Beaker

The image runs on Beaker as a custom Docker image (see the TL;DR spec above). Use
**`command: [python, /usr/local/bin/train.py, --flags…]`** (the TL;DR form): `train.py` sets the run
env from its `--flags` — including the identity defaults `MODEL_SIZE=32b`, the bf16 SFT script, sink on,
the deepseek tokenizer — then execs `bootstrap.sh`, so the rendezvous shim still runs.

> ⚠️ Do **NOT** omit `command:` and try to configure via `envVars` alone unless you set the *full*
> identity set — `MODEL_SIZE=32b`, `SFT_SCRIPT_NAME=Olmo-3-32B-SFT-bf16.py`, `SEQ_LEN`, `OLMO_CP_STYLE`,
> `OLMO_USE_SINK=1`, `OLMO_HF_TOKENIZER=1`, `DATASET_HF`, plus every tuning knob. With the bare ENTRYPOINT
> and none of these, `run.sh` falls back to its **7B / 65536 / ring** defaults and silently trains the
> wrong model at the wrong context. The `command: train.py` form gives you all of them for free — prefer it.
>
> `command: train.py --flags` needs the flags **baked** in the image you pull. The current pushed image
> has them all; **`docker pull` the latest `cu128-fa2-sink` first.** (Recipe *logic* — run.sh, the sft
> script — is cloned at runtime, so only new `train.py --flags` ever need a fresh image.)

`bootstrap.sh` auto-maps Beaker's rendezvous env, so **you set nothing extra for multi-node**:

| Beaker injects | mapped to |
|---|---|
| `BEAKER_REPLICA_COUNT` | `WORLD_SIZE` (#nodes) |
| `BEAKER_REPLICA_RANK` | `GLOBAL_RANK` (node index) |
| `BEAKER_LEADER_REPLICA_HOSTNAME` | `MASTER_ADDR` |

Single-node = `resources.gpuCount: 8`. Multi-node = `replicas: N` + **`leaderSelection: true`** +
`hostNetworking: true` + `propagateFailure/Preemption: true` (leaderSelection is required or the leader
hostname is unset and torchrun can't rendezvous — `bootstrap.sh` warns if so). Put the run's knobs in the
`command: train.py --flags` list (TL;DR); any knob also has an `OLMO_*` env you can add to `envVars`.

- **Storage:** mount a WEKA bucket **read-write** at `/data/training` (`datasets: [{mountPath:
  /data/training, source: {weka: <bucket>}}]`) — that's where checkpoints land and persist. Don't route
  the ~251 GB distcp checkpoints through a `result` dataset. Nothing in the chain writes to `/results`
  (logs go to `/data/training/logs` on WEKA), so a `result.path` is optional and will be near-empty —
  omit it, or point it at a real artifact.
- **Resume:** the trainer auto-loads the latest checkpoint under `save_folder` on start
  (`load_strategy` + `maybe_load_checkpoint`). If Beaker's `timeout` (e.g. 48h) kills a still-training
  32B (the download + one-time HF→distcp convert + compile eat into the first window), just re-submit the
  **same** spec with the **same WEKA mount + `RUN_NAME`** — it picks up from the last ephemeral/persistent
  checkpoint. Give long runs a generous `timeout` or plan to resume.
- **Secrets:** `beaker secret write HF_TOKEN …`, then `envVars: [{name: HF_TOKEN, secret: HF_TOKEN}]`.
  For HF checkpoint auto-upload, the token needs **WRITE** scope.
- **Cluster:** target H100 (sm_90) — the image supports it. InfiniBand user-space libs are baked, so
  multi-node uses IB once the fabric env + device access are set (see *InfiniBand* in the TL;DR). The
  jupiter values (`NCCL_SOCKET_IFNAME=ib`, `NCCL_IB_HCA=^=mlx5_bond_0`) are **examples** — confirm against
  AI2's current cluster guidance; a wrong `NCCL_SOCKET_IFNAME` makes NCCL hang at init rather than fall back.
