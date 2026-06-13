# Olmo-3 32B SFT — Technical Report & Training Notes

*Consolidated technical record for the Olmo-3 32B `olmo_32b_fp8` SFT run.
Single source of truth for a tech report or blog post — every claim here is traced to code (`file:line`) and,
where possible, cross-checked against empirical observations (checkpoint size, throughput, grad-norm). Author:
training agent, 2026-06. Supporting deep-dives are indexed at the bottom.*

---

## 1. What the run is

Supervised fine-tuning of **`allenai/Olmo-3.1-32B-Think`** on a math-reasoning corpus, with **olmo-core**, on
**4 × 8 H200 (32 GPUs)** on an ABCI-class shared cluster (Singularity/Apptainer, shared Lustre). Target: an IMO-level math model
for an invite-only Kaggle competition, trained from a fully-open base.

**Model (verified `nn/transformer/config.py:725`):** 32.23 B params · 64 layers · d_model 5120 · 40 query
heads / 8 KV heads (GQA) · head_dim 128 · SwiGLU FFN hidden 27648 · vocab 100278 (dolma2) · untied lm_head ·
mixed **sliding-window (~¾ of layers) + periodic full attention** · **YaRN** rope_scaling factor 8 (8192→65536),
bit-identical to the base model's `config.json`.

## 2. Training configuration (as actually resolved)

Launched via the Fields wrapper `fields/train.py --experiment olmo_32b_fp8 --olmo-ac-budget 0.4` (no LR/batch/
epoch/seq overrides). The wrapper's argparse defaults + env exports drive `olmocore/run.sh` →
`olmocore/sft_scripts/Olmo-3-32B-SFT-local.py`. Resolved values:

| Knob | Value | Source |
|---|---|---|
| Global batch | **1,572,864 tokens** (1.5 M) | recipe `train.py:100` |
| Sequence length | **65,536** | recipe `train.py:78` |
| Context parallel (cp) | **4** (from max-tokens-per-rank 16384) | `Local.py:184-190` |
| Data-parallel world | **8** (32 GPUs / cp 4) | `Local.py:198` |
| Rank microbatch | **65,536 = 1 seq/rank** (required for cross-doc masking) | `Local.py:212` |
| Grad accumulation | **3** (1.572M / 8 / 65536) | `Local.py:220` |
| **Epochs / max_duration** | **1 epoch ≈ 24,096 steps** | recipe `default_epochs=1.0` `train.py:77` |
| Peak LR | **5e-5** (overrides the inert hardcoded 8e-5) | recipe `train.py:100`, merge `Local.py:699` |
| LR scheduler | **CosWithWarmup**, warmup_fraction **0.03**, alpha_f **0.1** (floor 5e-6) | `Local.py:646-648` |
| Optimizer | **SkipStepAdamW**, betas (0.9, 0.95), wd 0.0, eps 1e-8, max_grad_norm 1.0 | `Local.py:594,655` |
| Adam moment dtype | **bf16** (both m and v) | `OLMO_OPTIM_DTYPE=bf16` default `train.py:199` |
| Fused linear CE | **ON** | `OLMO_FUSED_LCE=1` default `train.py:196` |
| Parallelism | **HSDP** (shard_degree 2 intra-node), param bf16 / reduce fp32 | `Local.py:503-507` |
| Activation checkpointing | budget mode 0.4 | `--olmo-ac-budget` |
| Precision | **fp8 rowwise** (FFN linears only) | `OLMO_FP8=rowwise` `run.sh:159` |

## 3. FP8 mixed-precision design — the core methodology

**Key architectural fact (resolves a common confusion):** every transformer block has **two** sub-layers — an
attention sub-layer *and* a feed-forward (FFN) sub-layer. "Sliding-window vs full" describes only the
*attention* sub-layer; **every block also has an FFN**, and the FFN is the larger half.

FP8 (torchao `convert_to_float8_training`, in-place `nn.Linear`→`Float8Linear`) is applied to the
**feed-forward linears only** (`feed_forward.w1/w2/w3`). **All attention projections (q/k/v/o) are deliberately
excluded** and kept BF16, following the **DeepSeek-V3 recipe** (attention has the widest activation dynamic
range; V3 §3.3 keeps attention operators, embeddings, output head, gating, and norms at original precision).
`fp8_attention_ignores` (`Local.py:317`) computes the exclusion FQNs from a meta build; `lm_head.w_out` is
auto-excluded; embeddings/norms aren't `nn.Linear` so torchao never touches them.

**FP8 coverage is the *majority* of GEMM compute, not a minority:** FFN ≈ **425 M params/block** vs attention
projections ≈ **63 M/block** → **~87 % of linear (GEMM) FLOPs are fp8'd.** So "lots of fp8, little speedup" is a
*comm-bound* signature (§7), not absence of fp8.

### Per-component dtype table (verified)

| Component | dtype | Notes |
|---|---|---|
| FFN forward GEMM | **FP8 e4m3** | torchao rowwise `config.py:294` |
| FFN gradient GEMM | **FP8 e4m3** | rowwise casts grad_output to e4m3 too — **not e5m2** (that's the tensorwise default) |
| Attention projections q/k/v/o (windowed **and** full) | **BF16** | excluded, DeepSeek-V3 recipe |
| SDPA / flash-attn core (QK·, softmax, ·V) | **BF16** I/O, FP32 internal | flash_2 varlen |
| Embeddings, lm_head | BF16 compute / FP32 master | not fp8 |
| RMSNorm | FP32 internal compute | params fp32 master / bf16 compute |
| **Adam moments (m, v)** | **BF16** | `OLMO_OPTIM_DTYPE=bf16`; bf16 *second* moment is an aggressive memory opt |
| FSDP master weights | **FP32** | MixedPrecisionPolicy |
| Param all-gather / compute copy | BF16 | `param_dtype=bfloat16` |
| Gradient reduce-scatter | **FP32** | `reduce_dtype=float32` |
| Loss / cross-entropy / logits | **FP32** | |
| FP8 matmul tensor-core accumulation | **FP32** | `torch._scaled_mm` |

**FP32 components:** master weights · gradient reduce-scatter · RMSNorm internal · RoPE · loss/CE · fp8 matmul
accumulation · flash-attn internal softmax. **Not** the Adam moments (bf16) or the param compute copy (bf16).

**Scaling choice:** rowwise (forced on sm_90 H200) = native CUTLASS per-row scaling — more accurate / more
outlier-robust than tensorwise, but slower and **cannot do fp8 FSDP all-gather** (that's tensorwise-only). This
is a deliberate accuracy-over-throughput trade; see §7 for the consequence.

**Empirical confirmation:** distcp checkpoint ≈ **251 GB** = fp32 master (4) + bf16 m (2) + bf16 v (2) =
**8 B/param × 32.23 B = 258 GB**. fp32 moments would be 12 B/param = 387 GB; all-bf16 would be 193 GB. The
observed size independently proves bf16 moments + fp32 master. (HF deliverable = 64 GB bf16-model-only.)

## 4. Deviations from AI2's reference recipe (intentional vs. silent)

The run is driven by the **Fields wrapper**, whose defaults differ from AI2's native `experiments.sh`. Some
deviations are deliberate recipe choices; three are **silent wrapper overrides that contradict the SFT script's
own documented defaults** — worth flagging for reproducibility and for interpreting the loss curve.

**Intentional recipe choices:** fp8-rowwise (AI2: bf16) · seq_len 65536 (AI2: 32768) · global batch 1.5 M
(AI2: 1.0 M).

**Silent overrides (the wrapper turns these on/changes them vs. the SFT script's stated "default"):**
1. **bf16 Adam moments** — `train.py:199` defaults `OLMO_OPTIM_DTYPE=bf16`; AI2 = fp32. Halves optimizer
   memory + checkpoint; puts the Adam *second moment* (the 1/√v denominator) in bf16. Loss-parity unvalidated.
2. **Fused linear cross-entropy ON** — `train.py:196` defaults `OLMO_FUSED_LCE=1`; the SFT script documents this
   as "default OFF" and reports **~6.5 % higher loss + lower throughput vs. the materialized reference** (use
   only when memory-bound, e.g. 32B). Likely an intentional memory trade for the 32B's 100k-vocab logits at seq
   65536, **but a real contributor to both the elevated loss and the modest throughput.** Whether the 6.5 % is a
   true quality hit or a z-loss/reporting artifact is **unvalidated** — a flagged open question.
3. **Scheduler = Cosine + alpha_f 0.1** (floor 5e-6); AI2 = **Linear + alpha_f 0.0** (decays to 0). The SFT
   docstring's claim "scheduler … EXACTLY upstream" is false. The LR never anneals below 5e-6.

**Epochs:** recipe sets **1**; AI2 README uses **2**; the SFT script hardcodes 3 (inert — overridden to 1). At
1 epoch + the larger 1.5 M batch this is a deliberate-looking shape, but it is fewer epochs than the reference —
**confirm intent.**

## 5. Training dynamics — observed behaviour, explained

### 5.1 Why the 32B's training loss is *higher* than a 7B run on a harder proofs-only subset
Counter-intuitive but fully expected; **not** a bug (masking verified, §6). Drivers:
- **Token-perplexity ≠ solve-difficulty.** Under teacher forcing, hard olympiad *proof prose* is the most
  *prefix-predictable* content in the corpus (rigid LaTeX, formulaic "WLOG/∎" scaffolding); its genuine "key
  idea" forks are **sparse** and, given the teacher's prefix, low-entropy. The 32B's extra data — **tool-use
  (Python/code), numeric computation, short answers** — is *prefix-resistant* (you must compute the exact value;
  the prefix can't telegraph it) → higher per-token CE.
- **Distribution breadth.** The 32B trains 3 modalities (proofs 42 % / numeric 29 % / tool 29 %; judge+numeric ≈
  54–60 % of *trained* tokens) vs. the 7B's homogeneous proof style → higher entropy floor at 1 epoch.
- **fused-LCE** adds a documented ~6.5 % to the reported loss (§4), and may further confound the 7B-vs-32B
  comparison if the 7B ran without it.
- Both at **1 epoch** → no memorization advantage either way.
- **Verdict:** cross-dataset training-loss comparison is uninformative about model quality; judge on held-out
  math eval. Lower 7B loss ≠ better 7B.

### 5.2 Gradient norm ~½ the 7B run
Expected. Larger model → per-parameter gradients shrink (loss signal spread over ~4.5× more params; the raw
2× understates a ~5× per-param reduction once the √N param-count growth is accounted for). The fixed 1.5 M-token
batch also denoises the averaged gradient toward its smaller true norm. Well under the `max_grad_norm=1.0` clip,
so clipping rarely binds. A grad-norm *above* the 7B would have been the red flag; it isn't.

### 5.3 Grad spikes + SkipStepAdamW
Spikes occur at ~the 7B's frequency (reassuring for fp8 at scale) and are filtered by **SkipStepAdamW**: it skips
the whole optimizer step when the latest loss *or* grad-norm exceeds a rolling **mean + 6σ** band over a 128-step
window — keeping the anomalous gradient out of *both* the weights and the Adam moments (so a spike can't poison
`v` and suppress the effective LR for ~2000 steps, the way plain clipping would). Watch the *skip rate* (should
stay <1 %), not the spike presence. Skipped steps still advance the scheduler, so the step counter slightly
overstates applied updates.

### 5.4 LR schedule behaviour
Peak 5e-5 → 3 % linear warmup → cosine decay to the 5e-6 floor over 1 epoch (~24,096 steps). LR at fraction `p`
of training (p ≥ 0.03): `5e-5 · [0.1 + 0.45·(1 + cos(π·(p−0.03)/0.97))]`.

| Progress | LR | % of peak |
|---|---|---|
| 60 % | 2.14e-5 | 42.8 % |
| 70 % (≈ step 16,900) | **1.48e-5** | 29.6 % |
| 75 % | 1.20e-5 | 24.0 % |
| 100 % | 5.00e-6 | 10.0 % (floor) |

## 6. Data integrity & correctness (audited, clean)

- **Intra-document attention masking is ON.** `TokenizerConfig.dolma2()` → `bos_token_id=None`, `eos=100257`
  (NOT the `dolma2_sigdig()` variant which sets bos=100257). With bos=None, `get_document_lengths` splits at
  every `<|endoftext|>` → correct per-conversation `doc_lens` → `cu_doc_lens` → **flash-attn varlen mask + RoPE
  intra-doc position reset** (`transformer/model.py:402-406`, `rope.py:533`). CP uses **ring/llama3**
  (doc-mask-aware), selected because `generate_doc_lengths=True`; the Ulysses path (which had a cu_doc_lens OOB
  assert) is opt-in and unused. **No cross-attention between packed conversations.**
  - Note on mechanism: flash-attn cu_seqlens are *derived from* `doc_lens` — they are not an independent guard.
    A `bos=100257` bug would corrupt `doc_lens` and flash-attn would faithfully cross-attend. We're safe because
    the upstream boundary detection is correct, not because flash-attn is immune.
- **Label masking** is `1=trainable / 0=masked → −100`, loss on trainable tokens only (~10 % masked for this
  dataset; mechanism is dataset-agnostic).
- **YaRN** rope_scaling (factor 8, attention_factor 1.20794…, original 8192) is bit-identical to the base
  model's config — no long-context corruption.
- **Dataset** is pre-tokenized dolma2, packed to 65536, sha256-verified against HF (300/300 files).

## 7. Multi-node scaling — why ~10 days, not the linear-extrapolated 7

A 1-node 8×H200 run showed a ~28-day ETA; naive linear scaling predicts 28/4 = 7 days for 4 nodes. Actual ≈ **10
days** ⇒ **~2.8× speedup, ~70 % scaling efficiency.** The ~30 % loss is structural, not a misconfig:
- **Inter-node gradient sync.** On 1 node all FSDP/cp comm is intra-node; on 4 nodes HSDP replicates across nodes
  → a per-step gradient all-reduce **over the network** that didn't exist on 1 node. Rowwise fp8 forfeits the
  fp8 all-gather, so this traffic is full bf16 bytes.
- **Strong-scaling Amdahl.** Fixed 1.5 M batch ⇒ each GPU does ¼ the compute per optimizer step (grad-accum 3 vs
  12 on 1 node), but per-step overheads (the new all-reduce, optimizer step, skip-step stats, kernel launches)
  don't shrink → they become a larger fraction → sub-linear.

**Unifying insight:** at 4 nodes / seq 65536 the run is **communication-bound, not compute-bound.** This single
fact explains *both* the modest fp8 speedup (§3 — fp8 accelerates FFN GEMM *compute*, not comms) *and* the
sub-linear node scaling. To approach 7 days you'd need a fatter interconnect, tensorwise fp8 with fp8 all-gather
(halving comm bytes, at an accuracy cost), or a larger global batch — none free, none changeable mid-run. ~70 %
efficiency for a 32B at 64k context over 4 nodes is normal-to-good.

## 8. Submission strategy — the under-annealed-checkpoint problem

The competition deadline (~7 days) lands **before** the 1-epoch run finishes (~10 days), at ≈ **70 % progress,
LR ≈ 1.48e-5** (~3× the 5e-6 floor) — i.e. a **mid-anneal checkpoint** that hasn't been through the low-LR tail
where SFT quality consolidates. The day-7 checkpoint is a valid, competent model but leaves quality on the table
for sharp math reasoning. Options (re-targeting the live schedule is impossible — no job control):
- **Cooldown anneal** (e.g. rented Vast.ai 8×80GB): branch from the day-7 **weights only** (optimizer state not
  needed — fresh Adam + a short warmup absorbs the v=0 transient), **start at ~1.5e-5 and decay to ~0** (do *not*
  re-spike to peak), short horizon (~0.5–1.5 B tokens). Recovers much of the annealing benefit cheaply.
- **Checkpoint soup / SWA:** uniformly average the last several *uploaded* checkpoints (preserved on HF despite
  keep-last=1). Averaging late high-LR trajectory points approximates the basin-centering that annealing buys —
  zero extra GPU, do it on day 7 before quantizing. Only average late, consecutive (mode-connected) checkpoints;
  eval the soup vs. the best single checkpoint before submitting; average *before* quantizing.
- Then **quantize** for the Kaggle submission.

## 9. Open questions / flagged for validation
1. **Epochs = 1 intended?** (AI2 = 2.) Affects total tokens seen and the LR horizon.
2. **fused-LCE's ~6.5 % loss penalty** — real quality hit or reporting/z-loss artifact? Validate vs. materialized.
3. **bf16 second moment** (Adam `v`) — loss-parity vs. fp32 unvalidated.
4. **Could windowed-attention `o_proj` be fp8'd safely** to recover throughput? Empirical; rowwise is the right
   substrate; requires a short loss-parity A/B before committing. The field is moving toward fp8 attention
   (DeepSeek-V4 ships FP8 attention QAT), but V3 (which this recipe follows) keeps it bf16 conservatively.

## 10. Operations (summary)
- **HF Xet / DNS:** the cluster's compute nodes had *flaky DNS* to the HuggingFace Xet host
  `cas-bridge.xethub.hf.co` (intermittent name resolution, not a hard block). Fix: a retry-loop `hf download`
  with `HF_HUB_DOWNLOAD_TIMEOUT=30` (so a stalled lookup errors fast and the loop resumes) + `HF_HUB_DISABLE_XET=1`;
  pre-stage model+data on Lustre and launch with `--model_path`/`--dataset_path` to bypass the in-container download.
- **Disk/quota:** ABCI group area default is **32 TiB + 100 M inodes** (shared per group); check with `show_quota`
  on a login node (not `lfs`, not inside the container). The 32B distcp checkpoint ≈ 251 GB; with keep-last=1, the
  latest is always on disk and every persistent checkpoint also uploads to its own HF repo.

## 11. Index of supporting documents (this repo)
- `debugging/fp8-precision-audit.md` — fp8 dtype audit + adversarial verification (file:line).
- `debugging/32b-high-loss-suspect-bos_token_id.md` — bos/intra-doc masking investigation + RESOLUTION.

*Every quantitative claim above was verified against code (`file:line`) by a multi-agent adversarial audit and,
where applicable, cross-checked against empirical telemetry (251 GB checkpoint, ~10-day runtime, grad-norm,
throughput). Items in §9 are explicitly unvalidated and flagged as such.*
