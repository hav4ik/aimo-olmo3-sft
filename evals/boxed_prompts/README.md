# boxed_prompts

Prompt templates for `\boxed{}`-answer math problems, **byte-exact with the training distribution**
of the Olmo-3 Thinking SFT dataset (`chankhavu/smolmo-sft-olmocore-pretokenized`).

## Files
- **`boxed_cot_prompt.py`** — no-tool **CoT solution** prompt (system + user). `build_messages(problem)`.
- **`tool_boxed_prompt.py`** — **tool-use** `\boxed{}` solution (Python exec). `build_tool_messages(problem, ...)`.
- **`cascade2_prompts.py`** — **proof / evaluation / analysis** (no-tool). See section below.
- **`tool_proof_prompt.py`** — tool-augmented proof variant.
- **`refine_prompt.py`** — single-turn **self-refine** (model improves its OWN draft against the 0/0.5/1 rubric).
- **`bestofn_prompt.py`** — **best-of-N selection** (pick the best of N candidate solutions). See section below.

Every builder returns chat `messages`; feed to the Olmo-3 chat template with `add_generation_prompt=True`.

## Provenance (measured on source `math_notool`, Cascade2)
- SYSTEM prompt: 100% uniform (also identical across `math_v4_cot`, `math_v4_tir_nc`):
  `You are an expert mathematical assistant. Provide rigorous, complete proofs. You are not allowed to use tools.`
- USER instruction (PREPENDED, blank line, then problem), 6,362/7,386 rows:
  `Please reason step by step, and put your final answer within \boxed{{}}.`
- The double brace `\boxed{{}}` is the exact trained form (an upstream `str.format` artifact). Build by
  concatenation only — do NOT run it through `str.format()` (would collapse `{{}}`→`{}`).

## Notes
- For the *tool-augmented* `\boxed{}` variant, the system prompt is the `<functions>` tool prompt and the
  instruction is appended (e.g. `Put your final answer in \boxed{}.`); not included here (this is the CoT/no-tool set).

## cascade2_prompts.py  (proof / evaluation / analysis — math_proof source)
- `build_proof_messages(problem, self_verify=False)` — proof solution. `self_verify=True` also self-grades 0/0.5/1.
- `build_evaluation_messages(problem, solution)` — evaluate a solution, score **0 / 0.5 / 1** in `\boxed{{}}`.
- `build_analysis_messages(problem, solution)` — analyze a solution, grade **0 / 1 / 6 / 7** in `\boxed{{}}`.
- Same SYSTEM prompt as the boxed/no-tool set. All templates verified by reconstructing **22,980** real
  training messages byte-exact (0 mismatches). Build by concatenation only (boilerplate holds literal `\boxed{{}}`).

## bestofn_prompt.py  (best-of-N candidate selection — the "judge / pick best" task)
The model gets a problem + an enumerated list of N candidate solutions in ONE user turn, and selects the best
by emitting a bare **`Judgment: <idx>`** (0-based, NOT boxed). It does not rewrite/merge candidates — output
is a choice, not a new solution. Distinct from `refine_prompt.py` (improves the model's *own* draft) and from
evaluation/analysis (grade a *single* solution).
- `build_bestofn_messages(problem, solutions, with_tools=True)` — `with_tools=True` = `math_withtool` (tool
  system prompt + `functions`; judge may run Python); `False` = `math_notool` (no-tool proof system prompt).
- `parse_judgment(output)` — pull the selected index from an assistant completion.
- **Prevalence (full dataset, all 1024 shards / 2,813,055 convs):** **228,692 prompts = 8.13%** (~1 in 12).
  By source: `math_withtool` 190,089 / `math_notool` 38,603.
- **N:** range 2..23, **mean 7.4, median 7**; N=2 is the mode (22.6%); ~84% have N≤12.
- **Candidates are ~always boxed:** 99.6% of 1,700,303 candidates contain `\boxed{}` (97.6% of prompts have
  ALL boxed). Feed boxed candidates to stay on-distribution.
- Verified byte-exact (user + system incl. `functions`) on **5,558** real training prompts — 0 mismatches.
