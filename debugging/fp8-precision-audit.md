# FP8 precision audit — Olmo-3 32B SFT (`--experiment olmo_32b_fp8`)

**Question:** is the run actually fp8, or silently bf16? And what is the exact dtype of each component?
**Method:** static code audit (`fields/train.py`, `olmocore/run.sh`, `olmocore/sft_scripts/Olmo-3-32B-SFT-local.py`,
`OLMo-core/src/olmo_core/{float8,nn,optim,train}`) + 3 adversarial sub-agents + an empirical checkpoint-size cross-check.
**Verdict:** **fp8 is genuinely active** (rowwise, feed-forward linears only). Not silently bf16. The modest,
bf16-looking throughput is the expected ceiling of *FFN-only + rowwise + no-fp8-allgather* on a multi-node
long-context run — by design, not a misconfiguration.

## Enablement chain (verified, no silent-bf16 path)
`olmo_32b_fp8` → `PRECISION=fp8` (train.py:100,314) → sm_90 ⇒ `OLMO_FP8=rowwise` (run.sh:159) → SFT builds
`Float8Config(ao_recipe=AOFloat8LinearRecipe["rowwise"], modules_to_ignore=attn_projs)` (SFT:567) → passed to
train module (SFT:641) → `apply_fp8` runs **unconditionally** before cp/compile/FSDP
(`train/train_module/transformer/common.py:60-62`, logs `"Swapped linear layers to Float8 linear layers"`) →
real torchao `convert_to_float8_training` in-place swap (`float8/__init__.py:146`). A bad ignore list **raises**
(`float8/__init__.py:152`) — fail-loud, never silent. Adversarial agent could not find any silent-bf16 path.

## Per-component dtype table

| Component | Actual dtype | User's reference | Match | Evidence |
|---|---|---|---|---|
| Attention projections `w_q/w_k/w_v/w_out` — **windowed AND full layers** | **BF16** (excluded from fp8) | BF16 | ✅ | `fp8_attention_ignores` matches every `blocks.<i>.attention` → SFT:317-330,551 |
| SDPA / flash-attn core (QK·, softmax, ·V) | **BF16** I/O, FP32 internal softmax/accum | BF16 | ✅ | flash_2 varlen, `nn/attention/__init__.py:487-517` |
| FFN (feed-forward) FORWARD GEMM | **FP8 e4m3** | FP8 e4m3 | ✅ | torchao rowwise `config.py:294-302` |
| FFN gradient (Dgrad/Wgrad) GEMM | **FP8 e4m3** (rowwise casts grad_output to e4m3, **not e5m2**) | FP8 e5m2 | ❌ | torchao `config.py:300-302`, docstring:119 — e5m2 is the *tensorwise/default* split, never used on rowwise |
| Attention gradients | **BF16** (attn excluded from fp8) | BF16 | ✅ | follows from exclusion above |
| Adam moments `exp_avg` (m), `exp_avg_sq` (v) | **BF16** | BF16 | ✅ | `OLMO_OPTIM_DTYPE` defaults to **bf16 via train.py:199**, exported :343 → `_opt_dt=bf16` SFT:586 → `SkipStepAdamW(dtype=bf16)` SFT:595 |
| Embeddings + lm_head | BF16 compute / FP32 master; not fp8 | — | — | `nn.Embedding` (not Linear); `lm_head.w_out` auto-excluded (model.py:591) |
| RMSNorm | FP32 internal compute; params FP32 master / BF16 compute | — | — | `nn/layer_norm.py:221-236` (`x.float()`) |
| FSDP master weights | **FP32** | — | — | factory `dtype=float32`; MixedPrecisionPolicy `param_dtype=bf16`, `reduce_dtype=fp32` (SFT:503-509) |
| Param all-gather / compute copy | BF16 | — | — | MixedPrecisionPolicy `param_dtype=bfloat16` |
| Loss / cross-entropy / logits | **FP32** | — | — | materialized LCE `logits.float()` `cross_entropy_loss.py:35` |
| FP8 matmul accumulation (tensor-core) | **FP32** | — | — | `torch._scaled_mm` accumulates fp32 |

### Two divergences from the original reference
1. **Gradient GEMM = e4m3, not e5m2.** The fwd-e4m3 / grad-e5m2 split is torchao's *tensorwise/default* behavior.
   **Rowwise** (forced on H200) overrides grad_output to **e4m3** for all three operands, so e5m2 is never used.
2. *(initially mis-stated as a mismatch, then corrected)* **Adam moments ARE bf16** — the original reference was right.
   The first audit pass looked at AI2's `experiments.sh` (no `OLMO_OPTIM_DTYPE` → fp32); the real launch path is
   `fields/train.py`, which **defaults `--olmo-optim-dtype` to `bf16`**.

## Which components are FP32
FSDP **master weights** · gradient reduce-scatter (`reduce_dtype=fp32`) · RMSNorm internal compute · RoPE
(`rope.py:514,672`) · loss/cross-entropy/logits · fp8 matmul tensor-core accumulation · flash-attn internal
softmax/PV accumulation · optimizer step counter.
*(NOT the Adam moments — those are bf16. NOT the param compute copy — that's bf16.)*

## Empirical cross-check — the checkpoint size proves the moment dtype
A 32B distcp checkpoint stores **master + Adam m + Adam v**:
- fp32 master (4) + **fp32** m (4) + **fp32** v (4) = 12 B/param = **384 GB**
- fp32 master (4) + **bf16** m (2) + **bf16** v (2) = 8 B/param = **256 GB ≈ observed ~251 GB ✅**
- bf16 master (2) + bf16 m (2) + bf16 v (2) = 6 B/param = 192 GB (too small)

⇒ observed ~250 GB ⇒ **fp32 master + bf16 moments**, confirming the table independently of the code.

## Why throughput looks bf16-ish (expected, not a bug)
1. **FP8 covers feed-forward linears ONLY** — all attention q/k/v/o projections are deliberately excluded
   (DeepSeek-V3 recipe: attention operators kept at higher precision). A large share of GEMM FLOPs stays bf16.
2. **Rowwise is the slow fp8 variant** on H200: native CUTLASS per-row-scaled kernels and **no fp8 FSDP
   all-gather** (tensorwise-only). At 4-node/32-GPU the comm-bound portion gets zero fp8 speedup.
3. At seq 65536 / cp=4 / ac 0.4, wall-clock is dominated by bf16 attention + bf16 comms + AC recompute; the fp8
   FFN GEMMs are a modest slice. Net win is single-digit-to-low-double-digit %, easily mistaken for bf16.

## DeepSeek recipe basis (the reason attention is excluded)
- **V3** keeps *attention operators* (+ embedding, output head, MoE gating, norms) at **BF16/FP32**; fp8 is the bulk
  MoE-FFN GEMMs. V3 uses MLA (dense) — **no windowed attention**, so "windowed vs full" is an Olmo concept.
- **V3.2** adds DeepSeek Sparse Attention whose **lightning indexer runs in fp8** — i.e. moves *toward* fp8 attention.
- **V4** training stack lists **FP8 attention QAT** — explicitly bringing attention into fp8.
- Our olmo-core impl excludes **all** attention projections uniformly (windowed + full) — faithful to the V3 spirit.
Sources: arxiv 2412.19437 (V3), 2512.02556 (V3.2), lmsys.org/blog/2026-04-25-deepseek-v4.

## Runtime confirmation (the uploaded logs don't capture rank-0 stdout)
The `[olmocore] … fp8=rowwise …` banner + `"Swapped linear layers to Float8 linear layers"` live in rank-0's
olmocore stdout, which is **0 bytes in the uploaded log dataset**. To confirm on the live job, on the rank-0 node
grep the actual olmocore log for `Swapped linear layers to Float8 linear layers` and
`Ignored modules for Float8 conversion` (should list the `…attention.w_*` projections), or check W&B.
