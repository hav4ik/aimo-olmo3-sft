# Cascade-2 → Olmo-3 tool-format conversion

How we convert `nvidia/Nemotron-Cascade-2` tool-use messages (`math_tool`) into **Olmo-3's
native tool-calling format**, verified parseable by vLLM's shipped Olmo-3 parser. This was
built carefully (two adversarial-council rounds) because a formatting error here would make
tool-use un-parseable at inference and waste an expensive training run.

Converter: `scripts/cascade_to_olmo.py` (lib `convert_messages()` + `validate()` gate + parquet CLI).

---

## 1. The two formats

### Cascade-2 (source, XML, everything inline as string `content`)
- **system**: `# Tools …\n<tools>\n<function>\n<name>…</name>\n<description>…</description>\n<parameters><parameter><name>…</name><type>…</type><description>…</description></parameter><required>[…]</required></parameters>\n</function>\n</tools>` + free-text format instructions.
- **assistant call**: `<think>…</think>\n<tool_call>\n<function=NAME>\n<parameter=P>\nVALUE\n</parameter>\n</function>\n</tool_call>`
- **tool result** — TWO encodings (both occur in math_tool):
  - (a) `role:"user"`, content `<tool_response>\nOUT\n</tool_response>`
  - (b) `role:"tool"`, content **double-wrapped** `<|im_start|>user\n<tool_response>\nOUT\n</tool_response>\n` (~12% of tool results)

### Olmo-3 native (target — from `chat_template.jinja` + `allenai/Dolci-*-Tool-Use` + vLLM `Olmo3PythonicToolParser`)
- **system**: `content` + a `functions` field = `json.dumps([{"type":"function","function":{"name","description","parameters":<json-schema>}}])`. Template renders `…content <functions>{functions}</functions>`.
- **assistant call**: `content` = reasoning (`<think>…</think>`), `function_calls` field = **PYTHONIC** calls `name(arg=<python-literal>, …)`, **one call per physical line**. Template renders `…content<function_calls>{function_calls}</function_calls>`.
- **tool result**: a message with role **`environment`**; template renders `<|im_start|>environment\n{content}<|im_end|>`.

### Special tokens (verified in the Olmo-3 tokenizer)
**Only `<|im_start|>`(100264), `<|im_end|>`(100265), `<|endoftext|>`(100257) are special (single) tokens.** `<functions>`, `<function_calls>`, `</function_calls>`, `<think>`, etc. are **regular subword text** (3–4 tokens each); `added_tokens_decoder` is empty. Consequence: `skip_special_tokens=True` at inference does **not** strip the tool tags, so the text parser still finds them.

---

## 2. Conversion rules (`convert_messages`)
- **system** → `content` = canonical prompt below + `functions` = JSON-encoded `<tools>`.
- any message whose content is a **tool result** (boundary-anchored `<tool_response>` check, **any role** incl. `tool`) → `role:"environment"`, content = inner output (leading framing `\n` stripped; trailing `\n` **kept** — it's masked context and plausibly the tool's real newline).
- **assistant**: content = text before the first valid `<tool_call>`; `function_calls` = `\n`-joined `name(P=repr(VALUE))`.
- the real problem `user` (no `<tool_response>`) stays `user`.

**System prompt** (mirrors the previous dataset's "expert mathematical assistant" identity; "Provide rigorous, complete solutions." parallels the non-tool prompt's "…proofs."; see `SYSTEM_PROMPTS.md`):
> You are an expert mathematical assistant. Provide rigorous, complete solutions. You are provided with function signatures within `<functions></functions>` XML tags. You may call one or more functions to assist with the user query. Output any function calls within `<function_calls></function_calls>` XML tags. Don't make assumptions about what values to plug into functions.

---

## 3. Validation gate (`validate`) — quarantine, don't corrupt
Per row, only rows that provably round-trip + parse are shipped; the rest go to a quarantine parquet. Quarantine criteria:
- **pre-guards** (cheap, before O(n²) regex): `oversized_content` (>2 MB), `too_many_tool_calls` (>200; legit ≤~99), `multiple_tool_responses` (>1 block in one result msg → avoid greedy-merge).
- **role mapping**: all roles ∈ {system,user,assistant,environment}; no non-environment message retains `<tool_response>`; #tool-results == #environment; #assistant preserved.
- **`unbalanced_tool_tags`**: `<parameter=`/`</parameter>`, `<function=`/`</function>`, `<tool_call>`/`</tool_call>` counts must balance (catches a `code` value containing `</parameter>` or `</function></tool_call>` that would silently truncate).
- **`sentinel_in_value`**: any extracted arg value still containing a protocol tag → reject (catches the *balanced-nested* truncation that the byte round-trip can't, since converter and gate share `_CALLP`).
- **`parser_reject_or_timeout`**: each `function_calls` is parsed with the **exact** vLLM `TOOL_CALL_REGEX` (via the `regex` module, **1 s timeout** → catches ReDoS) + `ast` + `literal_eval`; the recovered `(name,{param:val})` is byte-compared to the original. None/mismatch → reject.

Observed quarantine rate ≈ **0.005–0.02%** (almost all `leftover_tool_response`, a conservative catch of a prose mention).

---

## 4. How it was validated (high-stakes → adversarial)
- **Council round 1** (4 agents, head samples): byte-exact round-trips; passed — but the head rows only had encoding (a).
- **Council round 2** (4 agents, diverse full-file samples incl. 80–99-call sessions + `role:tool`) found **3 real bugs**: (1) 🔴 `role:tool` results silently dropped (template has no `tool` branch) — my big check had missed it; (2) `code` containing `</parameter>`/`</function></tool_call>` truncates silently; (3) ReDoS in the parser regex (1 s) on `code` with many `kw=val,`. **All fixed** (role routing by content; tag-balance + sentinel guard; regex-timeout gate + DoS pre-guards) and targeted-verified.
- **Special-token / end-to-end check**: render → tokenize → decode → parse round-trips losslessly; `skip_special_tokens` keeps the tool tags; the vLLM-style parser recovers **every** call (maxcalls 11/11, longcode 31/31, manycalls 66/66; and 532,581/532,581 across a 92K-row scan).

---

## 5. Serving (vLLM = gold standard)
```
vllm serve allenai/Olmo-3-7B-Think \
  --enable-auto-tool-choice \
  --tool-call-parser olmo3 \
  --reasoning-parser olmo3
```
- Use **`olmo3`** (a dedicated alias), NOT `pythonic`. `--enable-auto-tool-choice` is mandatory. Chat template auto-loads (matches ours). vLLM tolerates both Python (`None/True/False`) and JSON (`null/true/false`) literals.
- ⚠️ **Serving caveat (deploy-config, not a data issue):** the Olmo-3 chat template reads tool defs from a per-message **`functions`** field, NOT the OpenAI `tools` request field. To get tools into the prompt, inject them into a system message's `functions` (our converted rows already do this) or use a serving template that maps `tools`→`<functions>`. Otherwise the model is told "no functions" and won't call tools (cf. vLLM issue #32534).
- **SGLang**: no `olmo3`/pythonic-compatible parser today (its `pythonic` expects bracketed `[…]`, comma-separated, no `<function_calls>` unwrap). Would need a ~direct port of vLLM's `Olmo3PythonicToolParser`. Out of scope for now.

---

## 6. Usage
```bash
python scripts/cascade_to_olmo.py \
  --in /mnt/data/Nemotron-Cascade-2-SFT-Data/math_tool.parquet \
  --out tables/math_tool_olmo.parquet \
  --quarantine math_tool_quarantine.parquet
```
Output `messages` schema: `list<struct<role, content, functions, function_calls>>` (non-set fields are null; the template's `… is not none` checks handle that). Other columns (domain/source/generator) preserved.

---

## 7. `math_tool` dataset facts
- **2,267,447 rows** (the largest of the three Cascade math files), `domain=math_tool`, generator DeepSeek-V3.2.
- It is **tool-augmented NUMERIC-answer** math (solve with a stateful Python interpreter, `\boxed{}` answers) — NOT proofs. It also contains some **judge/grader** rows (no tool calls). So it's the least on-task file for a *proof* pilot; include only if we want code-interpreter tool-use.
- Tool: single `stateful_python_code_exec(code: string)`.
- Converted output: `tables/math_tool_olmo.parquet` (+ `math_tool_quarantine.parquet`). **Final kept/quarantine counts + token total: TBD** (fill after the full conversion + tokenization).
