# Batch size, microbatch, context parallelism & memory — OLMo-core

How global batch, per-device microbatch, context-parallel degree, and grad-accum
relate in our OLMo-core SFT runs; which env knobs control what; and why we run at
`seq_len=65536`. All of this is about **our** scripts (`olmocore/run.sh` +
`Olmo-3-{7B,32B}-SFT-local.py`); the recipe math lives in AI2's `BatchSizeConfig`,
which we drive via env/CLI overrides and never modify.

---

## 1. Why `seq_len=65536` is non-negotiable for this data

The recipe builds the dataset with `LongDocStrategy.truncate` (in
`build_sft_dataset`): a document longer than `seq_len` is **truncated mid-text, not
dropped**. So `seq_len` is not "how much we keep" — it's "where we amputate the
longest proofs." Our data (`chankhavu/smolmo-proofs-cot-sft`, 302,058 docs,
6.08B tokens) is heavily right-skewed:

| stat | tokens |
|---|---|
| mean | 20,144 |
| median (p50) | 15,450 |
| p90 / p95 / p99 | 44,141 / 53,744 / 63,058 |
| max | 65,533 |

The token mass lives in the long tail, not the median:

| docs longer than | % of docs | **% of all tokens** |
|---|---|---|
| 16,384 | 46.9% | 74.8% |
| 32,768 | 18.3% | **42.5%** |
| 49,152 | 7.2% | 20.5% |

What each cap would do (truncate strategy → truncated docs end mid-proof, no EOS):

| seq_len | docs truncated | tokens cut | verdict |
|---|---|---|---|
| 16,384 | 141,646 (47%) | 36.7% of corpus | catastrophic |
| 32,768 | 55,142 (18%) | 12.8% of corpus | corrupts the hardest 55k proofs |
| 49,152 | 21,854 (7%) | 2.8% | still damages the elite tail |
| **65,536** | **0** | **0%** | clean (max doc 65,533; `num_samples_skipped: 0`) |

We pre-filtered the source to ≤65,536, so **65536 is lossless**; anything smaller
poisons the longest, most valuable reasoning chains. This is a stronger reason than
"we'd lose samples" — truncate doesn't lose them, it teaches the model to stop
mid-argument. `seq_len=65536` is the default in `run.sh` and must stay.

**Packing.** The dataset is `NumpyPackedFSLDataset` with `generate_doc_lengths=True`
→ docs are concatenated into 65,536-token windows with **intra-document attention
masking** (the `llama3` CP variant, doc-aware) so packed docs don't attend across
each other. Median 15.4k into 65,536 windows ≈ ~4 docs/window; 92.5% of tokens are
trainable; 0 wasted. ~92,800 packed windows/epoch.

---

## 2. The identity that governs everything

OLMo-core enforces, every optimizer step:

```
global_batch_size  =  rank_microbatch_size  ×  dp_world_size  ×  grad_accum_steps
   (you set, G)         (you set)               (= world_size / cp)  (TRAINER DERIVES)
```

- **`global_batch_size` (G)** — total tokens per optimizer step, across all nodes and
  all grad-accum microsteps. This is the recipe invariant. **LR 5e-5 (7B) is tuned for
  G = 1,048,576** — change G and you are off-recipe.
- **`rank_microbatch_size`** — tokens per **DP rank** per microstep, in whole
  sequences (`% seq_len == 0`). 262144 = 4 × 65536 = "4 samples".
- **`dp_world_size = world_size / cp`** — number of independent data-parallel groups.
- **`grad_accum_steps`** — **you never set this.** The trainer recomputes it from G and
  `rank_microbatch_size` so the identity holds, *regardless of node count*. That is how
  "G stays 1M across all nodes and grad accumulations" works automatically.

---

## 3. The trap: `rank_microbatch_size` is per-**DP-rank**, not per-**device**

A "DP rank" is a whole **context-parallel group** of `cp` physical GPUs, because CP
splits each sequence along its length across those GPUs. Therefore:

```
per-device tokens (activation budget on one physical GPU)  =  rank_microbatch_size / cp
```

At `seq_len=65536` the default `cp = 4`, so **per-device = `RANK_MICROBATCH_TOKENS / 4`.**

This per-device number **is** what AI2 calls *"16384 tok/rank"* — it equals
`MAX_RANK_MICROBATCH_SIZE_TOKENS`, the per-physical-GPU activation cap "this config can
handle on an H100." A default (no-env) run maxes per-device out at exactly that cap
(rmb 65536 ÷ cp 4 = 16384).

Rewritten in per-device terms, **cp cancels** — each GPU just sees `G / world_size`
tokens per step, chopped into per-device-sized microbatches:

```
grad_accum_steps  =  G / (per_device_tokens × world_size)
                  =  G × cp / (rank_microbatch_size × world_size)
```

---

## 4. How `cp` is chosen (and the cap that sets it)

`cp` is **auto-derived** by `BatchSizeConfig`, not set directly. With cap
`MAX_RANK_MICROBATCH_SIZE_TOKENS = 16384`:

```
cp = smallest power of 2 such that  seq_len / cp ≤ cap
```

| seq_len | cap | → cp | per-device for 1 seq |
|---|---|---|---|
| 32768 | 16384 | 2 | 16384 |
| **65536** | **16384** | **4** | **16384** |
| 65536 | 32768 | 2 | 32768 |
| 65536 | 65536 | 1 | 65536 |

So the cap does two things at once: it picks `cp` **and** ceilings the per-device
microbatch at the cap. Raising the cap lowers `cp` (less ring-attention comm, more
contiguous activation per GPU) and raises the per-device ceiling. **There is no env
knob for the cap yet** — see §8.

---

## 5. The knobs (env vars on `docker run -e`)

| Env var | Controls | Default | Wired in |
|---|---|---|---|
| `GLOBAL_BATCH_SIZE` | global tokens/batch **G** | 1,048,576 (7B) / 4,194,304 (32B) | `run.sh` |
| `RANK_MICROBATCH_TOKENS` | per-DP-rank microbatch (per-device = ÷ cp) | auto (capped → per-device 16384) | `run.sh` |
| `SEQ_LEN` | sequence/window length | 65536 | `run.sh` |
| `LR` | learning rate | 5e-5 (7B) / 1e-4 (32B) | `run.sh` |
| `EPOCHS` / `MAX_STEPS` | duration | 2 epochs | `run.sh` |
| `MAX_RANK_MICROBATCH_SIZE_TOKENS` (cap → cp) | **not wired** | hardcoded 16384 | — (§8) |

`RANK_MICROBATCH_TOKENS` appends `--train_module.rank_microbatch_size=<v>`, which
overrides the capped auto-value (merge happens after `BatchSizeConfig`); the trainer
then recomputes grad-accum from G. Off by default ⇒ a no-env run is AI2 bit-for-bit.

---

## 6. Worked examples — 4× RTX 6000 PRO (world=4, cp=4), G = 1,048,576

| want per-device | set `RANK_MICROBATCH_TOKENS` (= ×cp) | → grad-accum | per-GPU activation |
|---|---|---|---|
| 16,384 (default) | 65536 | 16 | 1× baseline |
| 32,768 | 131072 | 8 | 2× |
| 65,536 | 262144 | 4 | 4× |

```bash
-e GLOBAL_BATCH_SIZE=1048576 \    # G (leave at recipe default)
-e RANK_MICROBATCH_TOKENS=262144  # per-device = 262144/cp = 65536, grad-accum → 4
```

Both knobs are independent — e.g. `GLOBAL_BATCH_SIZE=2097152
RANK_MICROBATCH_TOKENS=262144` → per-device 65536, grad-accum 8, G = 2M.

Multi-node keeps G fixed, grad-accum absorbs the node count (rmb 262144, G 1M):

| nodes | GPUs | cp | dp_world | grad-accum | G |
|---|---|---|---|---|---|
| 1 | 4 | 4 | 1 | 4 | 1M |
| 2 | 8 | 4 | 2 | 2 | 1M |
| 4 | 16 | 4 | 4 | 1 | 1M |
| 8 | 32 | 4 | 8 | 0.5 ✗ | — (raise G or lower rmb) |

---

## 7. Constraints (violate → assert at startup, before any training)

- `RANK_MICROBATCH_TOKENS` must be a **multiple of `SEQ_LEN`** (whole sequences).
- `G` must be divisible by `RANK_MICROBATCH_TOKENS × dp_world_size` → grad-accum a
  positive integer. (per-device 49,152 → rmb 196,608 does **not** divide 1M evenly;
  use power-of-2 multiples, or bump G.)
- Keep `G` a clean (power-of-2) multiple of `SEQ_LEN × dp_world_size` so AI2's own
  `BatchSizeConfig` self-check passes before the override applies.
- `world_size` and `seq_len` must be powers of 2 (`BatchSizeConfig` asserts this).
- **Grad-accum is sequential**, so peak VRAM is set by **one** microstep
  (= per-device tokens), not by G. Bigger `RANK_MICROBATCH_TOKENS` = more VRAM +
  fewer/larger microsteps (better kernel utilization); G is free of memory effects.

---

## 8. Not yet wired: the `cp` lever (the cap knob)

`GLOBAL_BATCH_SIZE` + `RANK_MICROBATCH_TOKENS` give full control of **G** and
**per-device microbatch**, but `cp` stays 4 (so per-device only comes in multiples of
`seq_len/cp = 16384`, and the ring-attention comm pattern is fixed). To also control
`cp` — e.g. cp=2 so each GPU holds 2 contiguous full sequences (32768) with half the
ring comm, or cp=1 (no CP) if it fits — expose `MAX_RANK_MICROBATCH_SIZE_TOKENS` as an
env knob feeding `model_overrides`/the cap. We added this once
(`OLMO_MAX_RANK_TOKENS`, commit 3f97552) and reverted it (869a777) when we believed
65536 fit at cp=4; it's worth re-adding for the H200/Blackwell tier, where the 16384
cap ("for H100") leaves headroom. Keep it **off by default** so no-env stays AI2
bit-for-bit. Re-adding it gives the full triple — **G, per-device microbatch, cp** —
independently controllable.

---

## 9. Memory & run budget (7B, full-FT, fp32 Adam)

- **Model+optim+grad ≈ 112 GB**, sharded across the flattened `dp_shard × cp` mesh =
  **all** GPUs. On 4 GPUs ≈ **28 GB/GPU** fixed; on 8 ≈ 14 GB/GPU. (FSDP shards
  params/optim; CP shards activations — DP alone is throughput only, not memory.)
- **Activations** scale ~linearly with **per-device tokens** (with selective AC on
  `blocks.*.feed_forward`). Default per-device 16384 is light; 65536 is 4×. On the
  96 GB RTX 6000 there is large headroom over the 28 GB fixed cost — read the real
  number off the `gpu_monitor` W&B metric / `nvidia-smi` and tune up to taste. Watch
  step 0 for OOM; step down `RANK_MICROBATCH_TOKENS` if needed.
- **Steps**: G = 1M, seq 65536, 2 epochs → ~5,800 steps/epoch → **~11,600 steps**;
  warmup 3% ≈ 348. Checkpoints every 1,000 (persistent, distcp ~100 GB each → ~1.1 TB
  over a full run, **not auto-pruned** — raise `save_interval` or prune for the 1 TB
  budget); ephemeral every 500.

---

## 10. Quick reference

```bash
# baseline (AI2 recipe, default cp=4, per-device 16384):
-e EXPERIMENT=7b_bf16_cot

# spend the RTX 6000 / H200 VRAM on throughput (per-device 65536, G unchanged):
-e EXPERIMENT=7b_bf16_cot -e RANK_MICROBATCH_TOKENS=262144

# bigger global batch too (G = 2M; re-tune LR if you do this):
-e EXPERIMENT=7b_bf16_cot -e GLOBAL_BATCH_SIZE=2097152 -e RANK_MICROBATCH_TOKENS=262144
```

W&B startup log prints `rank_microbatch_size` and the derived grad-accum — verify
they read what you intended before letting a long run proceed.
