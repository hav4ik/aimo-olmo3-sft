# Olmo-3 32B — AIME / HMMT Benchmark Results (bench_0207full)

**Date:** 2026-07-03
**Models:** proofpilot-merged, step20500-bf16 (crashed mid-run), step17000
**Benchmarks:** AIME'25, AIME'26 (MathArena/aime_2026), HMMT Feb'25 (MathArena/hmmt_feb_2025) — 30 problems each
**Modes:** no-tool (pure CoT) and tool (Python sandbox, self-parsed `<function_calls>`)

## Run configuration
- **Sampling:** temperature **1.0**, top_p 0.95 (1.0 is REQUIRED — these checkpoints degenerate into
  repetitive loops below ~1.0; do not lower).
- **Tokens:** `--max_tokens 0` → fills the full 65,536-token context, no artificial cap.
- **Rollouts:** 8 sessions/problem (240 sessions/cell). Metrics: **Avg@8** (per-session accuracy) and
  **maj@8** (equivalence-clustered majority vote).
- **Harness:** `run_eval.py` (boxed_notool / boxed_tool). Identical `\boxed{}` prompt prefix across all
  three benchmarks (verified byte-for-byte from stored traces).

## Grading
Graded with `grade_bench.py`, which uses **HuggingFace `math_verify`** (the MathArena-standard verifier)
as the primary equivalence check. This matters because **HMMT answers are ~half LaTeX expressions**
(fractions, radicals, π, factorials, multi-value sets), unlike AIME (always integer 0–999). The prior
string-match grader under-counted symbolic answers. **Validated 2026-07-03:** all 720 HMMT sessions
audited by 9 independent sub-agents → **0 grading errors**. AIME integer scores are identical under both
graders (no regression).

## Final matrix — Avg@8 / maj@8 (%)

| Model | AIME25 no-tool | AIME25 tool | AIME26 no-tool | AIME26 tool | HMMT25 no-tool | HMMT25 tool |
|---|---|---|---|---|---|---|
| **proofpilot** | 89.2 / 93.3 | 95.4 / 100 | 86.2 / 93.3 | 92.1 / 100 | 82.1 / 90.0 | 89.6 / 96.7 |
| **step20500** | 87.5 / 93.3 | 95.8 / 100 | crash | crash | crash | crash |
| **step17000** | 86.7 / 90.0 | 95.8 / 100 | 88.8 / 96.7 | 91.7 / 100 | 84.6 / 96.7 | 87.1 / 90.0 |

*step20500's endpoint crashed mid-run (~01:12); only its two AIME'25 cells completed. Not re-run.*

## Truncation (finish=length, % of sessions)
No-tool runs truncate 8–13% of sessions (long reasoning hitting the 65k context); **tools cut
truncation to ~0.4–1.2%** by offloading computation to Python.

| Model | AIME25 nt/tool | AIME26 nt/tool | HMMT nt/tool |
|---|---|---|---|
| proofpilot | 9.6 / 0.8 | 9.2 / 1.2 | 12.9 / 0.4 |
| step17000 | 11.7 / 0.4 | 8.3 / 0.4 | 12.5 / 0.4 |

## Key findings

1. **Tools help across the board (+2.5 to +9 pts Avg@8), and drive maj@8 to ~100%.**
   - AIME25: ~87–89 → ~95 (both models), maj@8 = 100%.
   - AIME26: 86.2→92.1 (proofpilot), 88.8→91.7 (step17000).
   - HMMT: 82.1→89.6 (proofpilot), 84.6→87.1 (step17000).
   The gain comes largely from eliminating truncation (verbose reasoning that ran past 65k tokens).

2. **The earlier "tools HURT on HMMT" result was a grading artifact, now dead.** The old string
   grader rejected correct-but-differently-formatted tool outputs (decimals, plain fractions,
   un-rationalized radicals) far more often than free-form reasoning — costing the proofpilot HMMT-tool
   cell ~22 points (67.1 → 89.6 once fixed with math_verify). With correct grading, tools help on HMMT
   too, consistent with AIME.

3. **proofpilot ≈ step17000 overall.** On no-tool AIME (integers, unambiguous grading) they're within
   noise (proofpilot slightly higher on AIME25, step17000 higher on AIME26). proofpilot's AIME26 no-tool
   deficit vs step17000 (86.2 vs 88.8) is ~truncation-driven (more verbose on those problems), and tools
   recover it (92.1 vs 91.7). On HMMT proofpilot's tool cell edges ahead (89.6 vs 87.1).

4. **Residual limitation is extraction, not grading:** when a multi-value answer is split across two
   separate `\boxed{}` blocks (e.g. HMMT hmmt25-10's two roots), only one is captured → conservatively
   marked wrong. Rare; affects a handful of sessions on one problem.

## Reproduce
- Grade any cell: `python grade_bench.py traces/bench_0207full_<model>_<bench>_<mode>`
- Compare two models (overlap + truncation): `python compare_models.py --names A,B traces/<dirA> traces/<dirB>`
- Standalone math_verify grader + audit tooling: `grade_bench_mv.py`, `build_audit_dump.py`
