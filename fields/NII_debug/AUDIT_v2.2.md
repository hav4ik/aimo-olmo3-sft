# v2.2 pre-ship audit — findings & actionable items

**Date:** 2026-06-08. **Branch:** `olmo-sft-32b` (v2.2 candidate) vs shipped `nii-attempt2` (v2.1).
**Method:** 8 parallel adversarial sub-agents, each given a distinct, non-overlapping slice and told to
assume a bug exists and read the actual code (cite file:line). Motivation: the operator runs this
**multi-node**, which we **cannot test** (VastAI is single-node) — so multi-node correctness had to be
established by reading, not running.

## Verdict
The **$20k-critical items are all CORRECT** (RoPE/YaRN, effective LR, topology, batch math, optimizer,
precision, loss). No BLOCKER/HIGH bug on the production code path. The real work is: one doc that
contradicts the code (fixed), and a handful of **32B-specific robustness** items (the run we can't test).

---

## ✅ Verified CORRECT (confidence record — re-checked line by line)
| Area | Verdict | Evidence |
|---|---|---|
| **RoPE / YaRN (training + shipped config)** | EXACT match to official Olmo-3 | factor 8, θ 500000, original 8192, β 32/1, attention_factor 1.20794 (=0.1·ln8+1), applied to full-attn layers only (not SWA), in both train graph and exported `config.json`. See [ROPE_YARN_AUDIT.md](ROPE_YARN_AUDIT.md). |
| **Effective LR** | **5e-5** for both models | The hardcoded `lr=8e-05` in the SFT scripts is a DEAD default; `run.sh:263` always passes `--train_module.optim.lr=5e-5`, which wins via `.merge(overrides)`. Verified the full path. |
| **SkipStepAdamW** | correct | `step_increment_bugfix=True` (default), betas (0.9,0.95), wd 0.0, eps 1e-8, max_grad_norm 1.0. |
| **Precision** | correct | fp32 master + bf16 param / fp32 reduce (FSDP2 MP); fp8 gated only by `OLMO_FP8` (set only when PRECISION=fp8); no cross-contamination. |
| **Loss / masking** | production-immune to the all-masked-NaN | `loss_div_factor` is batch-level, reused per microbatch → grad-accum can't reintroduce div-by-zero; only a fully-masked *rank-batch* NaNs, impossible for real SFT data. |
| **Multi-node topology** | correct | 4-var contract hard-required + validated; `unset` ordering safe; static rendezvous; `world=nodes×8`; non-pow2 traced clean into olmo-core's `build_world_mesh` (3 nodes → valid HSDP mesh shard2×replicate3). |
| **Batch math** | bit-identical on every 7B pow2 shape | 0 mismatches / 288 shapes; live 1.05M → grad_accum 8/4/2/1 on 1/2/4/8 nodes under both old & new code. |
| **run.sh / entrypoint.sh** | clean | `set -euo pipefail`; 4-var `:?` guards fire early; per-size dispatch correct (32B uses 32B script+arch, GBS 1.5M); exit-code propagates via `exec`; `bash -n` clean. |
| **Secrets** | clean | no committed token; `.gitignore` covers SECRETS.json / *.sif / *presigned*. |

---

## 🔧 Actionable items
| ID | Sev | Status | Area | Where | Action |
|---|---|---|---|---|---|
| **A1** | HIGH | ✅ FIXED 2026-06-08 | docs vs code | `MULTINODE.md`, `fields/README.md` | Docs said "pow2 only, 3 nodes rejected" — opposite of the code. Rewritten to the real rule (batch divisibility). |
| **A2** | HIGH | ⬜ OPEN | 32B convert robustness | `fields/upload.py` (~:179) | HF `<output>/model` written non-atomically & never cleared → a timeout/kill mid-convert leaves a corrupt dir the next poll may upload. Fix: `rmtree(hf_model)` before convert, or convert into `model.tmp-<pid>` then `os.replace`. |
| **A3** | MED | ⬜ OPEN | log observability | `fields/train.py` (watcher + `finally`) | `upload_run_logs` is only called *inside* the watcher loop; `stop.wait()→True` exits with no final flush, so the crash/shutdown log tail + final manifest never ship. Fix: one terminal flush after `watcher.join()` (hoist `log_dataset` out of its `if`). |
| **A4** | MED | ⬜ OPEN | batch-config generality | both SFT scripts (`microbatch_cap` block) | New microbatch-from-cap logic *rejects* a shape the old code accepted (`rank_batch < microbatch_cap`, e.g. B200 small-batch / short-seq). Production-safe (never hit at seq 65536/cp4). Fix is a provably-safe superset one-liner: `seqs_per_microbatch = max(1, min(microbatch_cap // seq, rank_batch // seq))`. |
| **A5** | HIGH (ops) | ⬜ OPEN | 32B disk | `olmocore/run.sh:51` + handoff | keep-last counts only *persistent* ckpts; +ephemeral +in-flight `-tmp` +HF copy → worst-case ~1.07 TB > 1 TB budget (same class as the 500 GB crash). Recommend **keep-last 1 for 32B** + size disk **≥1.2 TB**; fix the "~900 GB" comment. |
| **A6** | LOW | ⬜ OPEN | bash footgun | `run.sh:29` | `dirname ""` → `.` when libnvrtc is absent → writes `.` into ld.so.conf. Only triggers if libnvrtc missing (TE would fail anyway). Guard on the `find` result, not the dirname. |
| **A7** | LOW | ⬜ OPEN | config mislabel | export path | If someone overrides `SEQ_LEN` to a non-65536 value, shipped `max_position_embeddings` = that crop while YaRN still targets 65536 → internally-inconsistent (mislabeled) config. Cleaner: always ship `max_position_embeddings=65536`. Not reachable on the 65536 default. |
| **A8** | LOW | ⬜ OPEN | hardening | both SFT scripts (~`dp_world =`) | Add `assert world_size % cp == 0` for a clear local error (currently covered incidentally by the downstream shard-degree guard). |
| **A9** | MED | ⬜ OPEN | stale docs | `README.md:13`, `HANDOUTS.md`, `SCALEUP_32B.md`, `docker/base/README.md` | Still reference the removed axolotl image/Dockerfile. Purge from operator-facing docs. |
| **A10** | MED | ⬜ DEFERRED | versioning | `MULTINODE.md`, `fields/README.md` | Operator launch snippets name `olmo-sft-v2.1-allsm.sif`. Bump to v2.2 **when the v2.2 image is actually built** (not before). |

### Explicitly downgraded / NOT issues
- **Prune-vs-convert race** (an agent flagged MED): **not reachable** — convert is capped at 90 min, but 2 persistent saves span ~20 h (1000 steps × ~74 s/step) at the 32B's step time. No fix needed.
- Heterogeneous GPU-count nodes, MASTER_PORT collision message — NII allocates homogeneous nodes; standard torchrun behavior. LOW, no action.

---

## Design change shipped alongside (not an audit fix)
- **LR scheduler: linear → cosine, `alpha_f=0.1`** (both SFT scripts, warmup 0.03). Two deliberate
  deviations from AI2's stock SFT recipe (which is `LinearWithWarmup, alpha_f=0.0`):
  - **Cosine shape** — a gentler tail than linear (e.g. ~0.19·peak at 80% vs linear's 0.28·peak), so the
    model settles more smoothly and an early halt in the last ~20% is less risky.
  - **`alpha_f=0.1` floor** (5e-5 → **5e-6** final, not 0) — chosen because (1) the data is rich enough
    that we expect to keep improving to the end (don't want to kill the LR), and (2) we may model-merge
    (TIES / checkpoint soup), and a non-collapsed late trajectory keeps the endpoint in a broad, mergeable
    basin. NB: AI2's *pretraining* recipe uses cosine `alpha_f=0.1`; their *SFT* uses linear `alpha_f=0.0`.
    We're using a pretraining-style floor on SFT on purpose, for the two reasons above.
  - Applies to a *future* 7B re-run too (the live 7B was launched on linear `alpha_f=0.0` and is unaffected).

---

## 8-agent coverage map
1. Multi-node topology & rendezvous → CLEAR (LOW hardening only).
2. Batch / parallelism math → CLEAR on production; A4 (non-prod regression), A8 (latent-covered).
3. RoPE / YaRN → CONFIRMED CORRECT; A7 (non-default override only).
4. Checkpoint save/convert/upload/disk → **A2, A5** (the real 32B robustness items); prune-race downgraded.
5. Logging & HF log upload → **A3** (final-flush gap); node isolation / private / token / W&B-node0 all correct.
6. Optimizer / loop / LR / precision / loss → CLEAR; resolved the 8e-5-vs-5e-5 suspicion (5e-5 wins).
7. run.sh line-by-line → CLEAR; A6 (libnvrtc footgun).
8. Regression vs v2.1 + cross-cutting → **A1** (docs contradiction, fixed), A9 (axolotl doc refs), A10 (sif name); single-node path verified SAFE.
</content>
</invoke>
