# Audit history

This dataset's format was validated by repeated adversarial sub-agent councils (a formatting error
would make tool-use / reasoning un-parseable at inference and waste an expensive training run).
Summary of each round and its outcome.

## A. Cascade→Olmo tool converter (`cascade_to_olmo.py`) — 2 rounds
- **Round 1** (4 agents, head samples): byte-exact round-trips passed — but head rows only had one
  tool-result encoding.
- **Round 2** (4 agents, diverse full-file samples incl. 80–99-call sessions + `role:tool`) found
  **3 real bugs**: (1) 🔴 `role:tool` results silently dropped (template has no `tool` branch;
  ~12% of results are double-wrapped); (2) `code` containing `</parameter>`/`</function></tool_call>`
  truncates silently; (3) ReDoS in the parser regex. **All fixed** (route by content not role;
  tag-balance + sentinel guard; regex timeout + DoS pre-guards).
- **Gate**: a per-row `validate()` quarantines anything that doesn't provably round-trip through the
  real vLLM `Olmo3PythonicToolParser` (replica `_olmo_parse`). Observed quarantine ~0.01–0.02%.
  End-to-end: 532,581/532,581 calls recovered byte-exact across a 92K-row scan. See CASCADE_TO_OLMO.md.

## B. Final 3-set format audit (math_proof / no_tool / with_tool)
Pre-generation council confirmed, at scale: byte-exact system prompts, well-formed `<think>`
(816,585/816,585 math_proof assistant turns), 0 tool tags in non-tool sets, 100% vLLM tool-parse on
~30K with_tool calls. GO.

## C. v1 cross-reference (skeptical)
Audited the previous dataset (`smolmo-proofs-cot-sft`) as a reference, adversarially. Found 2
BLOCKER-class issues to AVOID: (1) `<think>` masking inconsistent between its olmocore (masked) and
axolotl (trained) artifacts; (2) a `<think>\n` boundary skew (the `>\n` fusion) — which led to the
`<think> ` space fix here. Also a baked "no functions" boilerplate contradiction and the `\boxed{{}}`
typo. All informed v2 decisions (see PROCESSING.md §6, INFERENCE_NOTES.md).

## D. Proofs-v2 parsing audit (2 agents)
- **Source**: full 17 GB scan — all 82,737 records `subset=proof`, `tools=[]`, roles only
  user/assistant, 0 `tool_calls`, 0 tool/environment, all have `reasoning_content`. Provably 100%
  non-tool → the "first user+assistant" parse is safe.
- **Output**: 0 tool tags/fields/roles; `<think>` format 0 violations / 82,737; 50/50 byte-exact
  round-trip (reasoning↔`<think>`, content↔answer); 1:1 conservation. GO.

## E. New-sources-vs-Cascade equivalence council (4 agents)
Differential audit: are the NEW sources (proofs_v2, math_v4_cot, math_v4_tir) processed IDENTICALLY
to the Cascade sources through the shared pipeline?
- **Think tokens** (agent 1): GO — `<think> ` open + `\n</think>\n\n` close + gen-prompt token-prefix
  + exact `num_tokens` (incl. the CAP_CHARS/truncation path) all identical new-vs-Cascade.
- **Tool format** (agent 2): GO — math_v4_tir == Cascade math_tool byte-for-byte (same OLMO_TOOL_SYSTEM,
  `functions` shape, pythonic one-call-per-line, `environment` role); 100% parse, despite different
  source formats (OpenAI `tool_calls` vs Cascade XML).
- **Pipeline/schema/system** (agent 3): GO — all sources flow through the same `process_rows`/
  `bake_nontool`/`keep_tool`/`record`/`SCHEMA`; system strings byte-identical; `fix_think` idempotent.
- **End-to-end differential** (agent 4): proofs_v2/math_v4_cot **format-indistinguishable** from
  Cascade; flagged 2 `math_v4_tir` items:
  - **DIFF 3a** ("`<think>` not closed before calls") — **FALSE ALARM** on re-check: `</think>` is
    never absent (0 cases); 99.4% sit immediately before `<function_calls>` (= Cascade), 0.6% have a
    valid one-line narration between. No fix.
  - **DIFF 3b** (no-call tir carries a tool system prompt) — **REAL**. 83% of `subset=tir` never call
    the tool. **Fixed**: re-split into `math_v4_tir` (real calls → tool system) and `math_v4_tir_nc`
    (no call → non-tool system, no `functions`). [user-confirmed]

## F. Deep-study council (4 agents — Olmo format, jsonls, tokenization, processing chain)
Final ground-truth-anchored study of the completed dataset (1024 shards, 2,813,055 rows, 37.90B tokens).

- **Olmo format vs OFFICIAL (agent 1): FAITHFUL.** Fetched real `allenai/Dolci-Think-SFT` + `Dolci-*-Tool-Use`
  samples. Our `<think> ` (space) open + `\n</think>\n\n` close match the official OpenThoughts-math
  sub-source exactly; tool format (`functions` in system, pythonic `function_calls`, `environment`
  results) matches official Dolci tool data AND the vLLM `olmo3` parser; special tokens, EOS, gen-prompt
  match. (Official Think mix is NOT standardized on open-whitespace — varies by sub-source; we match the math one.)
- **Original jsonls (agent 2): EXACT conservation, no silent loss.** All converter assumptions hold;
  no special-token leakage from source content. Notes (non-defects): dropped `expected_answer` (v4),
  FineProofs grade signals, problem URLs; Cascade math_tool/notool are dual-generator (DeepSeek-V3.2 +
  ~10–18% GPT-OSS-120B), ~10% of GPT-OSS rows dropped by the no-think filter (intended).
- **Tokenization (agent 3): CORRECT.** Only `<|endoftext|>`(100257)/`<|pad|>`(100277) runtime-special;
  reasoning/tool tags are regular subword text (skip_special_tokens-safe); no BOS; `num_tokens` ==
  full-tokenize training reality (0 mismatches); `<think> ` gen-prompt prefix + `\n</think>\n\n`
  (`198 524 27963 1363`) confirmed; CAP_CHARS+truncation overlong detection exact. Latent: literal
  `<|...|>` in content would inject special ids — none present in any source (data clean).
- **Processing chain (agent 4): GO.** Budgets unbiased, landed within ~1σ (notool 6.010B, withtool
  10.004B); ZERO data loss (2,813,055 rows conserved); single uniform 9-col / 4-field schema across all
  1024 shards; uniform full shuffle (no source clustering; every shard ~7–8 sources at global proportions).

### Minor items (non-blocking)
1. `fix_think` strips the opener space on EMPTY-reasoning turns → `<think>\n</think>` (no space) on
   ~1.3% of rows (intermediate tool-call turns with empty source `reasoning_content`). NEVER the first
   assistant turn, so the gen-prompt token-prefix invariant is intact. One-line fix for a future build.
2. Cross-source problem overlap: ~82k problems appear in >1 source (+ heavy multi-trace-per-problem).
   Mixture-design augmentation (different modalities); dedup intentionally skipped.
3. Dropped metadata (expected_answer / grades / URLs) could be materialized later for RL/eval.
4. Cosmetic: tool-call turns close `\n</think>` vs official `</think>` (1 token, parses identically).

**Consolidated verdict: GO.** The dataset is correct, faithful to the official Olmo-3-Think format,
exactly tokenized, losslessly conserved, and uniformly shuffled. No item blocks training.

## G. olmo-core training MASK + render verification (ran the real converter)
Ran the actual training-pipeline converter (`open-instruct/scripts/data/convert_sft_data_for_olmocore.py`,
`--chat_template_name olmo_thinker`) on diverse samples and inspected the emitted `token_ids` + `labels_mask`.

- **Mask (40 rows incl. a 100-tool-call conversation): GO.** 0 leaks. `environment` (tool-output) turns,
  `system`, `user`, and the `<|im_start|>assistant\n<think>` header+opener are MASKED; reasoning →
  `</think>` → answer → `<function_calls>…</function_calls>` → `<|endoftext|>` are TRAINED. The 100-call
  case: 100 environment blocks → 0 trained tokens; all 101 function_calls blocks trained. Per-source
  trainable fraction 0.80 (math_withtool, large masked tool outputs) … 0.998 (math_v4_cot). Mechanism:
  `mask_labels` masks every non-assistant role + extends the mask through the assistant generation header.
- **Render faithfulness (24 rows): GO.** The converter's decoded training text is BYTE-IDENTICAL to our
  shipped-template render (24/24, token-ids identical). open-instruct's `olmo_thinker` template == our
  shipped `chat_template.jinja` byte-for-byte; `add_bos=False` (no BOS); final token always `<|endoftext|>`;
  `<think> ` space open / `\n</think>\n\n` close / pythonic `<function_calls>` / `<functions>` in system /
  `environment` results all confirmed on the actual tokens; stored `num_tokens` matches 24/24.

Conclusion: the olmo-core pre-tokenization will train only on assistant content with tool outputs masked,
in exactly the verified Olmo-3-Think format. (This is what `prepare.sh` runs; the upcoming `olmocore/` build
just runs it over the full dataset.)
