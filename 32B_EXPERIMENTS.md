# 32B SFT — experiment log

Running log of every 32B training config we try, what worked, what didn't, and why. Append a row to
the table for each run. Model: `allenai/Olmo-3.1-32B-Think`, fp8 (torchao float8, Hopper flash_3),
`olmo3_32b` arch, SkipStepAdamW, HSDP, `OLMO_OPTIM_DTYPE=bf16` (bf16 Adam moments + fp32 master).
seq_len = **65536** for all runs (long-context, same as the 7B).

## How to read the memory math (so new configs are predictable)

**Sharding.** FSDP shards over the flattened `dp_cp = dp_shard × cp` mesh = **all GPUs in a node**
(`OLMo-core .../parallel/__init__.py:327-340` flattens `(dp_shard, cp)→dp_cp`). So on a single node the
32B is sharded across *every* GPU, regardless of how the per-rank token budget splits into shard vs cp.
- `--max-tokens-per-rank 16384` ⇒ `cp = ceil_pow2(seq / 16384)` (cp=4 at seq 65536; cp=2 at seq 32768).
  Either way per-rank tokens = 16384, and `dp_cp = GPUS_PER_NODE`.

**Peak memory is the OPTIMIZER STEP, not the forward.** Approx per-GPU state, sharded over `dp_cp` (= #GPUs/node):

| component | full-model bytes | ÷ dp_cp=4 | ÷ dp_cp=8 |
|---|---|---|---|
| params (bf16) | ~64 GB | 16 GB | 8 GB |
| grads (reduce_dtype fp32) | ~64–128 GB | 16–32 GB | 8–16 GB |
| optimizer (fp32 master + bf16 moments) | ~256 GB | 64 GB | 32 GB |
| AdamW step transients (`_foreach_div` etc.) | — | ~16–32 GB | ~8–16 GB |
| **peak state** | | **~112–140 GB** | **~56–72 GB** |
| + activations (per-rank 16384, AC-dependent) | | adds on top | adds on top |

H200 usable ≈ **139.79 GiB/GPU**. ⇒ **4×H200 (dp_cp=4) can't fit even at AC 0.0** (state alone ≈ peak).
**8×H200 (dp_cp=8) halves the state → ~30–50 GiB headroom → fits.** Activation budget (`--olmo-ac-budget`)
only moves the *forward/backward* peak; it can't shrink the optimizer-step peak, so it can't save 4×H200.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reclaims fragmentation ("reserved but unallocated")
— safe (allocator-only, bit-identical, no cudagraphs in our stack), recommended for the tight 32B run.

## Experiments

| # | date | GPUs | seq | max-tok/rank → cp | AC budget | expand_seg | result | failure point / peak |
|---|------|------|-----|-------------------|-----------|-----------|--------|----------------------|
| A | 06-08 | 4×H200 | 65536 | 16384 → cp4 | 0.0 | ? | ❌ OOM | (stage not captured) |
| B | 06-08 | 4×H200 | 32768 | 16384 → cp2 | 0.8 | no | ❌ OOM | forward (high AC = most VRAM) |
| C | 06-08 | 4×H200 | 32768 | 16384 → cp2 | 0.5 | no | ❌ OOM | **forward dry-run** — FFN buf `(16384, 27648)` fp8, 432 MiB short; 129.97 GiB allocated, 223 MiB free, **6.98 GiB fragmentation** |
| D | 06-08 | 4×H200 | 32768 | 16384 → cp2 | **0.0** | ? | ❌ OOM | **optimizer step** — `adamw.py:98 _foreach_div`, 68 MiB short; 136.84 GiB allocated, 40 MiB free, 220 MiB fragmentation |
| E | 06-09 | **8×H200** | per-rank 16384 | 16384 → (cp2/4) | **0.8** | no | ❌ OOM | **backward** — FFN activation `(16384, 27648)` fp8, 432 MiB short; 134.97 GiB allocated, 406 MiB free, 1.74 GiB fragmentation |
| **F** | 06-09 | **8×H200** | 65536 | 16384 → cp4 | **0.0** | no | ✅ **FITS** | **116 GiB/GPU = 82.5%**, ~24 GiB spare. **~74 s/step, ~14k tok/s (~1.75k/GPU)** → 6B tok / 5800 steps = **4d 23h** |
| **G** | 06-09 | **8×H200** | 65536 | 16384 → cp4 | **0.3** | **yes** | ✅ **FITS** | active **112 GiB (80%)**, reserved 115 GiB (82%); **1,842 TPS/dev (~73 s/step)**, 503 TFLOPS/dev, **ETA 4d18h**; loss 0.72 / PPL 2.05 / grad-norm 1.04 / 0 skips. global batch = default 1.05M |
| **H** | 06-09 | **8×H200** | 65536 | 16384 → cp4 | **0.5** | yes | ❌ OOM | **the cliff** — see below |
| **I** | 06-09 | **8×H200** | 65536 | 16384 → cp4 | **0.4** | yes | ✅ **FITS** | **85% / 118 GiB** — barely above 0.3 (82%). ⇒ **max fitting AC = 0.4** (cliff is 0.4→0.5) |
| **J** | 06-09 | **8×H200** | 65536 | 16384 → cp4 | **0.3** | yes | ⚠️ save ✅ / disk ❌ | ephemeral@10 steps: **async save validated** (251 GB, no GPU spike, convert ~7 min) then **DISK FULL at 500 GB** (two 251 GB ckpts + 64 GB export coexisted). Not a code bug — under-provisioned scratch. See operational guide below. |

**The AC→memory curve is a STEP FUNCTION, not a ramp.** 0.0→0.4 barely moves memory (82→85%); 0.4→0.5
jumps to OOM. Reason: the budget partitioner makes discrete keep/recompute decisions, and activations are
bimodal — many small ones (kept first, cheap) + the dominant **FFN intermediate `(16384, 27648)` fp8 =
432 MiB × ~64 layers**. Below ~0.45 it recomputes the FFN intermediates; at ~0.5 it flips them to *store*
across many layers → +20 GiB at once → OOM. So 0.4 is the max that fits, but throughput gain over 0.0/0.3
is tiny (weak AC lever) — **prefer 0.0/0.3 for headroom.**

### Notes per run
- **A/B/C/D (4×H200) → OOM, STATE-bound.** As AC dropped 0.8→0.5→0.0 the OOM moved *later* (forward →
  optimizer step). At AC 0.0 (the floor) it still OOMs at the AdamW update by ~68 MiB with fragmentation
  already minimal — **no knob left on 4 GPUs.** A 27–68 MiB margin isn't a stable run anyway (recompiles
  for dynamic `max_doc_len`, SkipStepAdamW skip-vs-step branches, doc-boundary variation all shift the peak).
- **E (8×H200, AC 0.8) → OOM, but ACTIVATION-bound, not state-bound.** It died in the **backward** on a
  stored activation, NOT at the optimizer step — so the *state now fits* (halved over 8 GPUs); the problem
  is AC 0.8 keeps too many activations. The 32B's layers are ~2× the 7B's (hidden ~7168, FFN 27648), so the
  7B's tuned AC 0.8 is far too generous here. **Fix = LOWER the AC budget** (0.0 = recompute everything =
  least VRAM). ⇒ next run: AC 0.0, then walk up to find the max that fits.
- **F (8×H200, AC 0.0) → FITS + first throughput baseline.** Command:
  `python train.py --experiment olmo_32b_fp8 --max-tokens-per-rank 16384 --seq-len 65536 --olmo-ac-budget 0.0 --run-suffix 8xh200test1ac00`
  (global batch = default 1.05M tok). **6B tokens = 5800 steps in 4d 23h** ⇒ **~74 s/step**, **~14k tok/s
  global** (~1.75k tok/s/GPU). This is the SLOWEST setting (AC 0.0 = full recompute) — raising AC to
  ~0.3–0.4 should cut step time ~15–25% (less recompute), and the warm compile cache cuts startup. So treat
  4d23h/6B as the *worst-case* throughput floor; tune AC up for the real run.
- **G (8×H200, AC 0.3 + expandable_segments) → FITS, but the AC lever gives DIMINISHING returns.** vs run F
  (AC 0.0): memory barely moved (active 112 vs ~116 GiB) and throughput barely improved (ETA 4d18h vs 4d23h,
  ~4% faster, ~73 vs ~74 s/step). So torch.compile's partitioner already recomputes cheaply — there's little
  to gain from raising AC, and the memory cost between 0.3 and the OOM at 0.8 is a cliff (not gradual). Implication:
  **AC 0.0–0.3 is the practical band**; 0.0 is the safest (most headroom) for ~4% slower. expandable_segments
  working (reserved−active gap only ~3 GiB). Training healthy (loss 0.72, 0 skipped steps). Note `MFU=161%` is an
  fp8-vs-bf16 reference-peak artifact — the real number is **503 TFLOPS/dev (~25% of H200 fp8 peak)**, which is fine.

## Checkpointing → convert → upload → disk (32B operational guide)

The full deliverable pipeline and its real, measured costs on 8×H200 (seq 65536, fp8). **This is the part
that bit us, so read it before a long run.**

### The pipeline (4 stages)
1. **distcp save** (olmo-core, on the training ranks, `save_async=True`): writes a sharded checkpoint to
   `…/output/internal/checkpoints/…/step<N>/model_and_optim`. **De-stages to CPU** (host RAM), then a
   background thread writes to disk — **no GPU memory spike** (verified: GPU stayed at 82% through the save;
   the per-rank `UserWarning: … ret.to(cpu_device)` lines are exactly that D2H de-stage — benign). Costs a
   one-time **~87 s GPU pause** per save (the de-stage), then training resumes.
2. **convert** (our `upload.py` → olmo-core `convert_checkpoint_to_hf.py`, **CPU-only subprocess**,
   `CUDA_VISIBLE_DEVICES=""`): reads the distcp, rebuilds the 32B on CPU, writes HF safetensors to
   `…/output/model`. **Measured ~7 min** (load `01:22:46` → shards written `01:29:24`; 2 shards, ~81 s write).
3. **upload** (same subprocess): pushes the HF model (~64 GB) to HuggingFace. **~7–45 min**, network-bound
   (depends on the box's uplink). Logs `uploading … -> hf://…` → `HuggingFace upload complete`.
4. **prune**: old checkpoints trimmed to `OLMO_KEEP_LAST_CKPTS`.

The watcher (node-0, background, CPU) runs steps 2–4 each poll on the **latest complete** checkpoint, deduped
by an `upload_successful.txt` marker. **It never touches the GPUs or slows training.**

### Measured sizes & times (8×H200, 32B fp8, seq 65536)
| item | size / time |
|---|---|
| **distcp checkpoint** (model + optimizer) | **251 GB** each |
| HF export (`…/output/model`, bf16, 2 shards) | **~64 GB** |
| convert (CPU) | **~7 min** |
| upload (64 GB, network) | **~7–45 min** |
| total ship time / checkpoint | **~15–50 min** |
| async-save GPU pause | ~87 s |
| step time (AC 0.0/0.3) | ~73 s |

### Failure mode 1 — DISK FULL (this crashed run J at 500 GB)
Each distcp is **251 GB**, and several big things coexist. Peak disk ≈
**`KEEP_LAST × 251` + one transient overlap (251, new save before old prune) + HF export (64) + base-model
artifacts (~64–128) + dataset`**. At the 10-step save interval, step10 (251) + step20 (251) + export (64) +
base (~64) ≈ **~630–700 GB > 500 GB → crash.**
- **Provision disk for `keep_last × 251 + 251 (overlap) + 64 (export)`** → **~880 GB at keep_last=2 ⇒ ~1 TB.**
- ABCI Lustre **group area** has the space; size the **quota** accordingly.

### Failure mode 2 — saving FASTER than we can ship (test-only artifact)
Ship time (~15–50 min) must be **< the save-interval wall-time**. At the **10-step** test interval
(~12 min/ckpt) we produced checkpoints faster than we shipped → backlog + disk fill. **The real run is nowhere
near this:** `--save-interval 500` ≈ **~10 h/ckpt ≫ 50 min ship** — the uploader is idle >95% of the time.
(Self-correcting anyway: the watcher ships only the *latest* checkpoint, so it skips intermediates rather than
queueing unboundedly; disk is bounded by `KEEP_LAST`, not upload speed.)

### Failure mode 3 — upload TIMEOUT kills it (and re-converts each retry)
The watcher bounds convert+upload with `FIELDS_UPLOAD_TIMEOUT` (**default 5400 s = 90 min**, `train.py:422`).
If exceeded it's **killed and retried next poll** — and the convert is **not resumable**, so each retry
re-pays the ~7 min convert → it can loop and **never land an intermediate checkpoint**. The 64 GB upload over
a slow uplink can push past 90 min.
- **Bump `FIELDS_UPLOAD_TIMEOUT=21600` (6 h).**
- The **final** upload (`train.py:739`, `timeout=None`) has **no cap** → the end-of-run model always ships,
  however long it takes. Only *intermediate* (watcher) uploads are at risk.
- Manual recovery if the watcher is stuck: `python /app/upload.py --output <output>` (run directly = no
  timeout) ships the already-converted model.

### Recommended production settings (32B)
- `--save-interval 500`–`1000` (≈10–20 h/ckpt) · `OLMO_KEEP_LAST_CKPTS 1`–`2`
- **disk ≈ 1 TB** (keep_last=2) · `FIELDS_UPLOAD_TIMEOUT=21600` · `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### Open items (code improvements offered)
- Make `FIELDS_UPLOAD_TIMEOUT` **size-aware** (default ~6 h for 32b) + **skip re-convert** if `…/output/model`
  already exists (so a retry doesn't re-pay the convert) — the real robustness fix for the slow-convert loop.
- ✅ **RoPE/YaRN — RESOLVED (training correct, warning benign).** Audited 2026-06-09 (see
  `fields/NII_debug/ROPE_YARN_AUDIT.md`). Training uses `YaRNRoPEScalingConfig(factor=8, beta_fast=32,
  beta_slow=1, old_context_len=8192)` + `rope_theta=500_000` — an **exact match** to the official
  `allenai/Olmo-3.{1-32B,7B}-Think` configs (factor 8 × 8192 = 65536). Weights are safe. The convert
  warning ("implicit factor -0.0001") is a **transient false alarm**: it's a `max_position_embeddings = -1`
  placeholder reloaded one line before it's patched to 65536; the FINAL `config.json` is correct, and our
  `upload.py` mirror doesn't touch the field. No fix needed for correctness (optional: a post-export assert).

## Conclusions so far
- ✅ **The 32B shards correctly** across all GPUs in a node (dp_cp). Forward fits at AC 0.0 even on 4×H200.
- ❌ **4×H200 is fundamentally too small** — STATE-bound; OOMs at the optimizer step even at AC 0.0.
- ✅ **8×H200 + AC 0.0 WORKS** (run F): ~116 GiB/GPU = **82.5%**, ~24 GiB spare — *more* margin than the 7B's
  ~94%, so it's a stable baseline. 8×H200 fits the state; high AC (0.8) is activation-bound → keep AC LOW.
- **AC budget for the 32B is LOW (~0.0–0.4), not the 7B's 0.8.** AC 0.0 = safe/slow (full recompute);
  ~0.3–0.4 = throughput sweet spot (keep peak <~90–95%); 0.8 OOMs.
- The compile cache + real fit run on **8×H200** at the production shape (seq 65536,
  `--max-tokens-per-rank 16384` → cp4, dp_cp=8). See `fields/NII_debug/H200_32B_PREBUILD.md`.

## To try next (8×H200)
- [x] **AC sweep done (8×H200, seq 65536, cp4):** 0.0 ✅82.5% · 0.3 ✅82% (4% faster, ETA 4d18h) · 0.5 ❌OOM · 0.8 ❌OOM. **Usable band = 0.0–0.3; max fitting = 0.3.** Cliff is sharp (partitioner keeps the big FFN activations >~0.4). **Ship AC 0.0 (most predictable/safest) or 0.3 (~4% faster, same ~82% peak).**
- [x] `expandable_segments` validated (run G) — works, ~3 GiB reserved−active gap.
- [ ] capture the 32B compile cache (per `H200_32B_PREBUILD.md`).
- [ ] confirm first distcp checkpoint save + HF upload (watch peak — the save adds some memory).
- [ ] decide lr/batch (AI2 32B = lr 1e-4 / batch 4.19M) before the real run.

## Open recipe question (not memory-related)
The olmocore 32B defaults are still the 7B's (`run.sh:58`: lr 5e-5, batch 1.05M). The AI2 32B recipe is
**lr 1e-4 / batch 4.19M tokens** (seq stays 65536 per our choice). Decide before a real run; irrelevant to
the fit test (override via flags; batch doesn't change peak memory, only step count).
