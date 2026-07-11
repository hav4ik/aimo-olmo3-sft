# System prompts (v2 mix) — per source

The redesigned dataset mixes tool-using and non-tool data, so the system prompt differs by
source. Baked at the **final-mix build** (system-prompt harmonization step), replacing each
row's original system message.

## Non-tool sources → `math_proof` (solutions + assessment) and `math_notool`
Exactly the previous dataset's prompt (`/mnt/data/aimo-proof-pilot/smolmo-proofs-cot-sft`,
"You are an expert mathematical assistant. Provide rigorous, complete proofs.") **+ an explicit
no-tools clause**:

```
You are an expert mathematical assistant. Provide rigorous, complete proofs. You are not allowed to use tools.
```

- Why: teaches the model to NOT emit tool calls when no functions are offered (prevents
  hallucinated `<function_calls>` in plain-reasoning contexts).
- Synergy with the Olmo template: a system message **without** a `functions` field renders with
  the appended boilerplate ` You do not currently have access to any functions. <functions></functions>`
  — so these rows *also* get the structural "no functions" signal automatically; the explicit
  sentence reinforces it in natural language.
- Applies to both `math_proof` (which IS proofs) and `math_notool` (numeric-answer) per the
  design decision — same identity across the non-tool corpus. (Note: `math_notool` is
  numeric-answer, not proofs, so "rigorous, complete proofs" is slightly off-task for it, but
  kept for a uniform non-tool persona.)

## Tool source → `math_tool`
The tool prompt baked by `scripts/cascade_to_olmo.py` (`OLMO_TOOL_SYSTEM`), with the tool schema
in the system message's `functions` field (rendered as `<functions>…</functions>`):

```
You are an expert mathematical assistant. Provide rigorous, complete solutions. You are provided with function signatures within <functions></functions> XML tags. You may call one or more functions to assist with the user query. Output any function calls within <function_calls></function_calls> XML tags. Don't make assumptions about what values to plug into functions.
```

- Parallel to the non-tool prompt: same "expert mathematical assistant" identity, and
  "Provide rigorous, complete **solutions**." (vs "proofs." for the non-tool/proof sources —
  math_tool is numeric-answer, so "solutions"), then the tool instructions instead of the no-tools clause.

## Summary
| source | system prompt | functions field |
|---|---|---|
| math_proof (solutions, assessment) | expert + "…not allowed to use tools." | none |
| math_notool | expert + "…not allowed to use tools." | none |
| math_tool | expert + "…rigorous, complete solutions." + Olmo tool instructions | tool JSON schema |

Implementation: the non-tool prompt is applied at the final-mix build (analogous to v1's
`apply_system.py`); the tool prompt is already in `tables/math_tool_olmo.parquet`.
