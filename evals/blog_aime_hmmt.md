# Benchmarking Olmo-3 32B on AIME & HMMT — With and Without Tools

We benchmarked our Olmo-3 32B checkpoints — `proofpilot-merged`, `step20500`, and `step17000` — on
three competition-math sets: **AIME 2025**, **AIME 2026**, and **HMMT February 2025**. Each was run in
two modes: pure chain-of-thought (**no-tool**) and with a **Python sandbox** the model can call. Below
are the numbers, plus a cautionary tale about grading that flipped one of our conclusions.

(`step20500`'s serving endpoint crashed partway through, so we only have complete numbers for AIME'25
and a partial AIME'26 no-tool result — reported below for what they're worth.)

## Setup

- **Sampling:** temperature **1.0**, top-p 0.95. (These checkpoints are unstable at lower
  temperatures — they fall into repetitive loops — so 1.0 is the production setting.)
- **Context:** full 65,536 tokens, no artificial output cap.
- **Rollouts:** 8 samples per problem (30 problems × 8 = 240 samples per cell).
- **Metrics:** **Avg@8** (average per-sample accuracy) and **maj@8** (majority vote across the 8 samples).
- **Prompt:** identical `\boxed{}` instruction across all three benchmarks.

## Results

**Avg@8 / maj@8 (%)**

| Model | AIME'25 no-tool | AIME'25 tool | AIME'26 no-tool | AIME'26 tool | HMMT'25 no-tool | HMMT'25 tool |
|---|---|---|---|---|---|---|
| **proofpilot** | 89.2 / 93.3 | 95.4 / 100 | 86.2 / 93.3 | 92.1 / 100 | 82.1 / 90.0 | 89.6 / 96.7 |
| **step20500** | 87.5 / 93.3 | 95.8 / 100 | 93.8 / 96.3 † | — crash — | — crash — | — crash — |
| **step17000** | 86.7 / 90.0 | 95.8 / 100 | 88.8 / 96.7 | 91.7 / 100 | 84.6 / 96.7 | 87.1 / 90.0 |

† `step20500` AIME'26 no-tool is **partial** — the endpoint crashed with only 27/30 problems and
211/240 samples collected. Since the easier problems tend to finish first, this figure is biased
upward and is **not** directly comparable to the fully-completed cells. The remaining three `step20500`
cells never ran.

## Takeaways

**1. Tools help across the board — and push majority-vote to near-perfect.**
Giving the model a Python sandbox adds **+2.5 to +9 points** of Avg@8 on every benchmark, and lifts
maj@8 to ~100% on both AIME years. On AIME'25 both checkpoints jump from ~87–89% to ~95%.

**2. Most of the gain comes from killing truncation.**
In no-tool mode, 8–13% of samples run past the 65k-token context and get cut off mid-reasoning. With
tools, the model offloads heavy computation to Python instead of grinding through it token-by-token,
and truncation drops to **~0.4–1.2%**.

| Truncation (% of samples) | AIME'25 nt→tool | AIME'26 nt→tool | HMMT'25 nt→tool |
|---|---|---|---|
| proofpilot | 9.6 → 0.8 | 9.2 → 1.2 | 12.9 → 0.4 |
| step17000 | 11.7 → 0.4 | 8.3 → 0.4 | 12.5 → 0.4 |

**3. The checkpoints are effectively tied.** On unambiguous integer-answer AIME, the two
fully-benchmarked models trade the lead within noise (proofpilot ahead on AIME'25, step17000 on AIME'26
no-tool), and wherever one trails on no-tool, tools close the gap. `step20500`'s completed AIME'25 cells
(87.5 / 95.8) sit right alongside the others.

## A grading gotcha worth sharing

Our first pass showed something bizarre: on HMMT, tools appeared to **hurt** proofpilot
(74.6% → 67.1%). That contradicted every other result.

The cause wasn't the model — it was the grader. Unlike AIME (always an integer 0–999), **half of
HMMT's answers are LaTeX expressions**: fractions, radicals, π, factorials like `2^{25}·26!`, even
multi-value sets. Our string-matching grader was rejecting correct answers written in a different but
equivalent form — and tool mode triggers this constantly, because Python tends to emit decimals, plain
fractions, or un-rationalized radicals (`9/√23`) instead of the textbook form (`9√23/23`).

Switching to HuggingFace's **`math_verify`** (the same verifier MathArena uses for these datasets)
fixed it. The corrections were large and concentrated on tool mode:

| HMMT cell | String grader | math_verify |
|---|---|---|
| proofpilot no-tool | 74.6 | **82.1** |
| proofpilot **tool** | 67.1 | **89.6** (+22.5) |
| step17000 no-tool | 76.7 | **84.6** |

AIME integer scores were **identical** under both graders — only the symbolic HMMT answers moved. To be
sure, we dumped all 720 HMMT samples and had a fleet of independent agents re-check every single
grade against the model's actual output: **zero grading errors**. `math_verify` correctly handles
`9/√23 = 9√23/23`, `√640 = 8√10`, `√(95/24) = √570/12`, reordered root-sets, and numeric-vs-symbolic
forms.

**The lesson:** if your benchmark has non-integer answers, exact-string matching will silently
understate your model — and it will penalize tool-augmented runs the most, precisely because tools
produce mathematically-correct output in machine-natural forms. Grade with a symbolic verifier.
