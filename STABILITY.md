# FP8 vs BF16 SFT — stability test harness

A 2×2: each framework gets a **BF16** arm (precision baseline) and an **FP8** arm,
**identical except precision**, so any loss/grad difference isolates FP8.

| | OLMo-core (reference) | Axolotl (fallback) |
|---|---|---|
| **BF16** | `launch/launch_7b_bf16.sh` | `runs/configs/olmo3-7b-bf16.yaml` |
| **FP8**  | `launch/launch_7b_fp8.sh`  | `runs/configs/olmo3-7b-fp8.yaml`  |

Recipe (all four): AI2 `7b_think_sft` — lr 5e-5, 2 epochs, global batch 1,048,576 tok,
seq 32768, olmo template, FA3 packed-varlen, auto-CP (cp_degree=2). `logging_steps: 1`
so per-step loss/grad-norm are visible.

## Run
```bash
# OLMo-core (8xH200, single node)
cd olmo3-olmocore-runs && NPROC_PER_NODE=8 ./launch/launch_7b_bf16.sh
                          NPROC_PER_NODE=8 ./launch/launch_7b_fp8.sh
# Axolotl (8xH100/H200)
cd olmo3-axolotl-sft && ./runs/scripts/launch.sh olmo3-7b-bf16 8
                        ./runs/scripts/launch.sh olmo3-7b-fp8  8
```
For a quick probe, cap steps (OLMo-core: `--trainer.max_duration.value=300 --trainer.max_duration.unit=steps`;
axolotl: add `max_steps: 300`) — ~300 steps is enough to see divergence.

## What to compare (stability signals)
1. **Loss-curve parity** — FP8 should track BF16 within ~1% (DeepSeek report <0.25%, Nemotron ~99% recovery). A widening gap or higher floor = instability.
2. **Grad-norm spikes / NaN-Inf** — watch `max_grad_norm` clipping frequency and any non-finite steps.
3. **OLMo-core SkipStepAdamW skip count** — it skips >6σ loss-spike steps; a *rising* skip rate under FP8 is the canary (it's masking FP8 noise).
4. **Throughput (tok/s)** — the FP8 payoff; confirm it's actually faster (FP8 without compile can be *slower* — both arms compile).
5. **Downstream eval** (if available) after a full run — the real test.

## Why BF16 is the baseline (prior art)
**All three reference labs do SFT/post-training in BF16**, reserving FP8 for pretraining
and/or inference:
- **AI2 Olmo 3**: SFT is bf16 (param bf16 / reduce fp32, no float8).
- **NVIDIA Nemotron Nano**: *"After post-training in BF16, we applied PTQ … to FP8"* (FP8 = inference only; pretrain used NVFP4/FP8 with attention + first/last layers high-precision).
- **DeepSeek-V3**: FP8 is a **pretraining** recipe; SFT/RL sections specify no FP8. (V4 adds FP4 *QAT* in post-training — quantization-aware, not plain FP8 SFT.)

So FP8 *training* for a short SFT is off the beaten path — this harness is to measure whether it's safe *here*, not to assume it.

## FP8 design notes (informed by DeepSeek-V3)
DeepSeek's FP8 recipe is **aggressive, not timid**: **all three Linear GEMMs — Fprop,
Dgrad (activation backward) AND Wgrad (weight backward) — run in FP8** (§3.3.1). What
protects it is (a) **fine-grained scaling** (1×128 activation tiles, 128×128 weight
blocks, online), (b) **FP32 accumulation** (FP8 tensor-core accum is only ~14-bit on
H800, so partials promote to CUDA-core FP32 every Nᴄ=128), and (c) a targeted high-precision
set — {embedding, output head, MoE gating, normalization, attention} at "BF16 or FP32",
**FP32 master weights + stored gradients, BF16 AdamW moments**. So the *compute path is
fully FP8 including both backward passes*; only the granularity, accumulation, and that
targeted set are protected. Result: **<0.25% loss gap vs BF16** (pretraining).

Mapping to our torchao setup (verified):
- **We match DeepSeek's compute aggressiveness**: `rowwise`/`tensorwise` cast input+weight+
  grad_output to FP8 → all three GEMMs (incl. Dgrad+Wgrad) in FP8, with FP32 `_scaled_mm`
  accumulation. (`rowwise_with_gw_hp` is the one variant that holds the weight-gradient GEMM
  higher-precision — a conservative lever if Wgrad-FP8 turns out to be the unstable part.)
- We are **coarser on granularity**: rowwise = per-output-channel (1-D rows), vs DeepSeek's
  2-D 1×128/128×128 blocks. Hopper torchao has no 2-D block scaling (that's ~MXFP8/Blackwell),
  so rowwise is the closest available approximation.
- **Attention matched to DeepSeek**: ALL attention projections (128 FQNs = 32 blocks ×
  q/k/v/o) are excluded from FP8 and stay BF16; FP8 covers the feed-forward linears only.

- **Granularity** is the key stability lever: `tensorwise` (per-tensor, coarsest, fastest)
  < `rowwise` (per-output-channel, native CUTLASS) < DeepSeek block < MXFP8 microscaling.
  **DeepSeek-V3 (<0.25% gap) AND Unsloth both find per-channel/block FP8 stable where
  per-tensor is risky** — so the OLMo-core FP8 arm now **defaults `rowwise`** (test viability
  with the safe recipe first); `OLMO_FP8=tensorwise` is the faster/coarser stretch to try
  only after rowwise is clean.
- **Axolotl FP8 has NO granularity knob** — it's accelerate's `tensorwise` (verified: the
  schema exposes only `fp8`/`fp8_enable_fsdp_float8_all_gather`), with accelerate-managed
  exclusions (no full-attention carve-out). So the **axolotl arm is the coarsest/riskiest by
  construction** — if FP8 is going to work anywhere it's the rowwise OLMo-core arm; axolotl
  tells you whether the *easy* FP8 path is also viable.
- **Accumulation & scaling**: torchao's `_scaled_mm` already accumulates in FP32 and scales
  online — the two tricks DeepSeek hand-rolled — so we get those for free.
- **Exclusions (DeepSeek-faithful)**: OLMo-core FP8 keeps **all attention + lm_head +
  embeddings + norms** in high precision (`fp8_attention_ignores`); FP8 is on the
  feed-forward linears only. The remaining gap vs DeepSeek is *granularity* (rowwise
  per-channel vs their 1×128/128×128 blocks) — if it's still unstable, `rowwise_with_gw_hp`
  (weight-gradient GEMM high-precision) is the next lever.

## Three FP8-*training* regimes, by risk (where the evidence sits)
1. **LoRA + frozen FP8 base** (Unsloth): base weights FP8 (used in rollout + forward),
   trainable adapters + gradients + optimizer all BF16, backward dequantizes to BF16.
   Unsloth reports SFT loss curves tracking BF16 here, and 1.4× faster vLLM rollouts.
   **Lowest risk — but it's LoRA, not the full-param SFT we run.** A genuinely-safe FP8
   *training* option if we ever want it (would be a separate config; say the word).
2. **Full-param FP8, rowwise** (≈ DeepSeek granularity) — our OLMo-core FP8 arm. Medium risk.
3. **Full-param FP8, tensorwise** — our axolotl FP8 arm (forced). Highest risk.

What our 2×2 actually probes is regimes 2–3 (full-param). Unsloth's "FP8 works" is regime 1
(LoRA frozen base) — encouraging, but it does **not** by itself validate full-param FP8 SFT;
that's what this test is for.

## Decision
If BF16 and FP8 loss curves overlap (within ~1%) with no spike/skip increase and a real
throughput win → FP8 is viable (prefer `rowwise` unless `tensorwise` proved clean). If not
→ ship BF16 (the reference choice) and save FP8 for inference PTQ.
