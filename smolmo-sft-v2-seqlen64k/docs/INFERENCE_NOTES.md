# Inference notes (v2) — for the dataset card & inference guide

Carry these into the HF dataset card and the eventual inference guide. We own the inference
pipeline, so we ship our own chat template (jinja) and need not rely on the model repo's stock one.

## 0. Authoritative template (verified against upstream)
We fetched `allenai/Olmo-3-7B-Think/chat_template.jinja` and ship it **verbatim** as
`chat_template.jinja` in this dataset. It differs from our earlier working copy in exactly ONE line —
the system-LESS default sentence (`"You are OLMo … built by Ai2 …"` vs `"You are a helpful AI
assistant."`) — which never fires because we always bake a system message. All functional branches
(assistant, `<function_calls>` rendering, per-message `functions`, `environment`, the
`<|im_start|>assistant\n<think>` gen-prompt) are byte-identical. Confirmed: the official Think template
renders tool calls as `<function_calls>{pythonic}</function_calls>` from the `function_calls` field —
exactly our converted format — and pre-opens `<think>` (token 29) in the gen-prompt.

⚠️ The **Think** template (unlike Instruct) has NO top-level `tools`/`tool_calls` support — it reads
tools ONLY from a per-message `functions` field. Serving the Think model with tools therefore requires
injecting them via a system/user message `functions` field (our `math_withtool` rows already do this);
vLLM `--tool-call-parser olmo3` then parses the `<function_calls>` output.

## 1. `<think> ` (space) reasoning opener — Dolma-aligned, NO template change 🟢
Our assistant turns open with **`<think> {reasoning}`** (a single SPACE after the tag), matching the
form Olmo was trained on in Dolma/Dolci (`<think> Okay, …`). The build transforms the upstream
Cascade/FineProofs `<think>\n…` → `<think> …` (`fix_think_open` in `scripts/build_mix.py`).

Why the space (not a newline): the Olmo-3 tokenizer **greedily merges `>` with a following newline**
(`<think>\n` → `[14023, 771, 397]`, `397`=`'>\n'`), but a following SPACE does NOT merge — `<think> Okay`
→ `[14023, 771, 29, 36539, …]` where `29`=`'>'` and `36539`=`' Okay'`. The **stock** Olmo template opens
generation with `<|im_start|>assistant\n<think>` ending at token `29`, so:

- `<think> ` (space) → stock gen-prompt is a **byte/token-EXACT prefix** of training; the model's first
  generated token is `' Okay'` (36539), exactly the loss target. **No custom template, no skew.** ✓
- `<think>\n` (newline) → `>` fuses to `397`; gen-prompt (ends `29`) is NOT a prefix → train/inference
  skew. ✗ (This is why we transform away from it.)

Verified: stock gen-prompt `[198,14023,771,29]` is an exact prefix of the space-form sequence; not of
the newline form. **vLLM/SGLang need no template change** — the shipped Olmo template already emits
`<think>` and the model generates the leading space itself. (vLLM `--reasoning-parser olmo3` searches
the literal `<think>`/`</think>` strings; the leading space is trimmed.) Applies to all three sets.

**Close marker (standardized to Dolci Think SFT form):** answer-bearing assistant turns close with
`{reasoning}\n</think>\n\n{answer}` — newline before, **blank line** after — matching Dolci. Tool-call
turns (no answer text; `<function_calls>` rendered next) close with `{reasoning}\n</think>`. The build
normalizes whitespace around `</think>` accordingly (`fix_think` in `build_mix.py`). Upstream was mostly
`\n</think>\n` (proof/notool 100%), `\n</think>`+call (tool), `</think>` no-ws (fineproofs) — now uniform.
Tokenization: `\n</think>\n\n` → `198('\n') 524('</') 27963('think') 1363('>\n\n')` then the answer.

## 2. Masking convention (olmo-core)
- We target **olmo-core** only (axolotl is not a concern for v2).
- The `<|im_start|>assistant\n<think>` opener is **masked** (it's prompt — emitted by the stock
  gen-prompt at inference). Loss starts at the first reasoning token `' Okay'` (36539), runs through
  `</think>`, the answer, and the terminating `<|endoftext|>`. The space form gives a **clean** mask
  boundary (token `29` masked | `36539` trained) — no straddling fused `>\n` token.
- `environment` (tool-output) turns are **masked** — context, not a target. (This is why the
  with_tool trainable fraction sits ~0.90 vs ~0.94–0.99 for the non-tool sets.)
- Decision: keeping the opener masked is fine since we enforce `<think>\n` at inference. (Training
  the opener would be marginally nicer but is not needed; not doing it.)

## 3. Turn terminators — `<|im_end|>` vs `<|endoftext|>` (positional, decided at render time)
The chat template terminates each assistant turn by position: `{% if not loop.last %}<|im_end|>{% else %}eos_token{% endif %}`
(eos = `<|endoftext|>`, 100257). So:
- A non-final assistant turn (e.g. a tool-CALL turn, or a prior round in a multi-turn chat) ends with
  `<|im_end|>` (100265). `environment` (tool-output) and user/system turns also end with `<|im_end|>`.
- Only the CURRENTLY-LAST assistant turn ends with `<|endoftext|>`.
This does NOT foreclose multi-turn: the terminator is recomputed on every render. When a conversation
grows and is re-rendered, the previously-final turn is no longer `loop.last`, so its `<|endoftext|>`
becomes `<|im_end|>`. The EOS is never stored in history — it's only the stop signal for the current
generation. (Host requirement: store assistant messages as TEXT and let the template re-render; don't
concatenate the literal stop token. Standard `apply_chat_template` loops do this.) `<|endoftext|>` thus
means "this complete response is done" while `<|im_end|>` means "turn done, conversation continues" —
the model learns to distinguish e.g. a tool-call (pause) from a final answer (stop).
- **Serving stops on BOTH tokens**: `allenai/Olmo-3-7B-Think` `generation_config.json` has
  `eos_token_id: [100265, 100257]` (`<|im_end|>` AND `<|endoftext|>`). The model often emits `<|im_end|>`
  at inference (it saw it on most turns); either halts generation. Verified against official Olmo
  multi-turn SFT data (Dolci tool-use 85% multi-turn, Think-SFT incl. 12-round chats): intermediate
  assistant turns → `<|im_end|>`, final → `<|endoftext|>`, 0 violations. See `AUDITS.md`.

## 4. Tool serving (with_tool only)
- vLLM: `--enable-auto-tool-choice --tool-call-parser olmo3 --reasoning-parser olmo3`.
- Tool defs live in a per-message `functions` field (rendered `<functions>…</functions>`), NOT the
  OpenAI `tools` request field — inject at serve time. See `CASCADE_TO_OLMO.md` §5.

## 5. `\boxed{{}}` artifact — intentionally kept
Some prompts carry the literal `\boxed{{}}` (un-escaped `str.format` double-brace from the upstream
Nemotron grading template). **We deliberately do NOT normalize it** — keeping odd bracket forms makes
the model more robust to sloppy user formatting at inference.

**Both empty forms coexist** — `\boxed{{}}` (double) and `\boxed{}` (single) appear side by side
across the corpus. The double-brace is an upstream **prep typo** (un-escaped `str.format` double-brace,
inconsistently introduced by different dataset authors), concentrated in the Nemotron analysis-grading
rubric + most math_notool answer-instructions — NOT a deliberate or universal format. The correct
single-brace `\boxed{}` is in fact more common overall. (Note: these counts are EMPTY instruction templates in prompts; the answers in
assistant turns are filled single-brace, e.g. `\boxed{42}`, regardless.)

Prevalence (scan `scripts/_boxct.py` / `_boxct2.py`; rows containing each empty form — a row may have both):
| table | scan | rows `\boxed{{}}` | rows `\boxed{}` | note |
|---|---|---|---|---|
| solutions | full 417,454 | 0 | 46,955 | uses correct single-brace only |
| assessment | full 399,131 | 198,010 | 144,737 | double = the 198,010 analysis/grading rows; single = evaluation rows |
| math_notool | sample 130k | ~88% | ~58% | mostly double instruction, often also a single elsewhere |
| math_tool_olmo | sample 26k | ~3.4% | ~82% | overwhelmingly single |

Because the model already sees BOTH `\boxed{}` and `\boxed{{}}` naturally, it learns to handle either —
exactly the inference robustness we want. Intentionally preserved (no normalization).
