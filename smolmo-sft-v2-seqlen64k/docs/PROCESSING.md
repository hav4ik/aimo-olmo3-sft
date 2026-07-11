# smolmo-math-cot-sft — processing documentation

End-to-end record of how this dataset was built: sources, conversion, the shared processing
pipeline, the Olmo-3 format conventions, the key decisions/findings, and reproduction steps.
Companion docs in this folder: `INFERENCE_NOTES.md` (serving + the `<think>` token nuances),
`CASCADE_TO_OLMO.md` (tool-format conversion + validation), `SOURCES.md` (provenance),
`SYSTEM_PROMPTS.md` (per-source system prompts), `AUDITS.md` (adversarial audit history),
`chat_template.jinja` (the verified upstream Olmo-3-7B-Think template, shipped verbatim).
All processing scripts are under `docs/scripts/`.

## 1. What this is
Math chain-of-thought SFT for **Olmo-3.1-Think (7B & 32B)**, in the model's native ChatML +
`<think>` format. Three task families — proof writing/grading, numeric-answer reasoning, and
tool-integrated reasoning (stateful Python) — mixed to a fixed token budget (~37.9B). Every
sample has a real `<think>` reasoning trace and is `< 65536` tokens. Final per-source tallies are
in the top-level `README.md` (written by the reshuffle step).

## 2. Sources & budgets
| source label | upstream | reasoning | budget | system |
|---|---|---|---|---|
| `math_proof` | Nemotron-Cascade-2 `math_proof` (solutions + evaluation + analysis) | DeepSeek-V3.x `<think>` | take-all (~13.9B) | non-tool |
| `fineproofs` | FineProofs-SFT (olympiad) | yes | take-all (~0.12B) | non-tool |
| `proofs_v2` | Nemotron-Math-Proofs-v2 | DeepSeek-V4 (`reasoning_content`) | take-all (~2.35B) | non-tool |
| `math_notool` | Nemotron-Cascade-2 `math_notool` | DeepSeek-V3.x `<think>` | **6B** (downsampled) | non-tool |
| `math_v4_cot` | Nemotron-SFT-Math-v4 `subset=cot` | DeepSeek-V4 (`reasoning_content`) | take-all (~3.16B) | non-tool |
| `math_withtool` | Nemotron-Cascade-2 `math_tool` | DeepSeek-V3.2 `<think>` | **10B** (downsampled) | tool |
| `math_v4_tir` | Nemotron-SFT-Math-v4 `subset=tir`, **with a tool call** | DeepSeek-V4 | take-all (~0.75B) | tool |
| `math_v4_tir_nc` | Nemotron-SFT-Math-v4 `subset=tir`, **no tool call** | DeepSeek-V4 | take-all (~1.61B) | **non-tool** |

Groups: proofs (no-tool) ≈16.4B · no-tool numeric ≈10.8B · tool ≈10.8B → **~37.9B total**.

## 3. Per-source conversion
All sources are normalized to one schema and one Olmo format; only the *ingest* differs.

- **Cascade `math_proof`** → split into `solutions` (task_type=solution) and `assessment`
  (evaluation + analysis, with parsed `score`/`score_normalized`). Already carry `<think>`.
  Tables: `tables/{solutions,assessment}.parquet`. (Built earlier in this workspace; see SOURCES.md.)
- **Cascade `math_notool`** → used directly (already `<think>` non-tool).
- **Cascade `math_tool`** → converted from Cascade XML tool format to **Olmo native** tool format
  by `scripts/cascade_to_olmo.py` → `tables/math_tool_olmo.parquet`. Full spec + validation in
  `CASCADE_TO_OLMO.md`.
- **FineProofs-SFT** → `reasoning_content` + `proof` assembled into `<think>…</think>{proof}`.
- **Nemotron-Math-Proofs-v2** (`scripts/convert_proofs_v2.py`) → each assistant message has
  `content` (final) + `reasoning_content` (the real DeepSeek-V4 trace). Assembled as
  `<think> {reasoning_content}\n</think>\n\n{content}`. 82,737 rows; 100% non-tool (audited).
- **Nemotron-SFT-Math-v4** (`scripts/convert_v4.py`) → **3-way split**:
  - `subset=cot` → `math_v4_cot` (non-tool, same assembly).
  - `subset=tir` **that actually calls a tool** (has a `tool` role or assistant `tool_calls`) →
    `math_v4_tir`. The OpenAI `tools` array → system `functions`; assistant `tool_calls` →
    pythonic `stateful_python_code_exec(code=…)` in `function_calls`; `reasoning_content` →
    `<think>`; `role:tool` → `role:environment`. Every call is validated through the real vLLM
    `olmo3` parser replica (`_olmo_parse`) and round-trip-checked; failures are quarantined (0 found).
  - `subset=tir` **with NO tool call** (83% of tir!) → `math_v4_tir_nc`: converted as **non-tool**
    (non-tool system, no `functions`) — a tool-less conversation must not advertise tools.

## 4. Shared pipeline (`scripts/build_mix.py`)
Every source flows through the identical processing (verified equivalent by the council, see
`AUDITS.md`):
1. **`has_think` filter** — drop any sample whose assistant turn(s) lack a well-formed `<think>`.
2. **System baking** — non-tool sources: `bake_nontool` replaces/prepends the single NONTOOL
   system prompt; tool sources: `keep_tool` preserves the baked `OLMO_TOOL_SYSTEM` + `functions`.
3. **`fix_think`** (idempotent) on every assistant turn → opener `"<think> "` (space, Dolma-style)
   and close `"\n</think>\n\n{answer}"` (answer turns) / `"\n</think>"` (tool-call turns).
4. **Exact length filter** — render through the Olmo template, tokenize, drop if `≥ 65536`. To
   avoid full BPE on monster reasoning traces, the length is computed on a `≤ 400k`-char prefix
   with `truncation=max_length=65536` (overlong detection stays exact; kept rows get the exact
   full count — verified 0 mismatches; a rare long-but-sparse row falls back to a full pass).
5. **Unified schema** — `messages` (list of `{role, content, functions, function_calls}`; non-tool
   rows have the latter two null), `source`, `num_tokens`, `task_type`, `score_normalized`,
   `generator`, `orig_source`, `problem_id`, `domain`.
6. **Budgeted reshuffle** (`scripts/reshuffle_budgeted.py`) — concatenate all source shards, random
   per-row keep to downsample `math_notool`→6B and `math_withtool`→10B (keeps the result fully
   shuffled, not front-loaded), take-all the rest, scatter into 128 buckets, per-bucket shuffle →
   uniform `data/train-XXXXX-of-NNNNN.parquet` (zstd) + `README.md`.

## 5. Format conventions (see INFERENCE_NOTES.md for the full rationale)
- Reasoning: `<think> {reasoning}` (single SPACE after the tag) … `\n</think>\n\n{answer}`. The
  space (not a newline) makes the **stock** Olmo gen-prompt `<|im_start|>assistant\n<think>` a
  byte/token-exact prefix of training → no train/inference skew, **no custom template needed**.
- Tools (Olmo native): system `functions` (OpenAI JSON array) → `<functions>…</functions>`;
  assistant `function_calls` = pythonic `name(arg=…)` one-per-line → `<function_calls>…</function_calls>`;
  tool results = role `environment`. Serve with vLLM `--reasoning-parser olmo3 --tool-call-parser olmo3`.
- Final assistant turn ends with `<|endoftext|>`. Only `<|im_start|>/<|im_end|>/<|endoftext|>/<|pad|>`
  are special tokens; the tool/think tags are regular subword text.
- Ship `chat_template.jinja` (= upstream allenai/Olmo-3-7B-Think, rev `7c991fde…`, verified identical
  to the validated copy except an unused system-less default line).

## 6. Key decisions & findings
- **`<think> ` space vs `<think>\n`**: the tokenizer fuses `>`+newline (→ one token), breaking the
  gen-prompt prefix; a space does not. We normalize to the space form (Dolma-aligned). [INFERENCE_NOTES §1]
- **`</think>\n\n` close**: matches Dolci Think SFT; standardized across all answer turns.
- **`\boxed{{}}`** (an upstream `str.format` double-brace typo) is **kept**, not normalized —
  the corpus already mixes `\boxed{}` and `\boxed{{}}`, so the model learns to tolerate sloppy brackets.
- **Judge / `Judgment:` rows** kept (faithful to source; not every sample ends in `\boxed{}`).
- **No-`<think>` rows dropped** (a small judge subpopulation in Cascade tool/notool).
- **V4 TIR no-call discovery**: 83% of `subset=tir` conversations never actually call the tool
  (tool was available, model answered directly). These are re-routed to the **non-tool** system
  (`math_v4_tir_nc`); only the 17% that truly call tools keep the tool system. [your call]
- **DeepSeek-V4 reasoning lives in `reasoning_content`** (a separate message field), not stripped —
  so Proofs-v2 / Math-v4 are first-class thinking data, not bare answers.

## 7. Reproduce (script order)
```
# tool-format conversion of Cascade math_tool
python scripts/cascade_to_olmo.py --in math_tool.parquet --out tables/math_tool_olmo.parquet ...
# new DeepSeek-V4 sources
python scripts/convert_proofs_v2.py        # -> tables/proofs_v2.parquet
python scripts/convert_v4.py               # -> tables/math_v4_{cot,tir,tir_nc}.parquet
# tokenize + filter + shard each source (take-all sources)
python scripts/build_mix.py                # Cascade math_proof/notool/tool + fineproofs
python scripts/process_new.py              # proofs_v2, math_v4_cot, math_v4_tir
python scripts/process_tir2.py             # math_v4_tir, math_v4_tir_nc (after the 3-way split)
# full shuffle + budgets -> data/ + README.md
python scripts/reshuffle_budgeted.py
```
(`recover_finalize.py` documents a one-off finalize recovery after a FineProofs shard-naming bug in
an earlier build run.) Run inside the `open-instruct-dataprep` image with the Olmo-3 tokenizer.

## 8. olmocore/
Pre-tokenized (packed) data for olmo-core training is generated **after** the `data/` parquet is
confirmed — into `../olmocore/` (raw uint32 `token_ids` memmaps + `labels_mask` + tokenizer +
stats; OBFD packing, seq_len 65536; mask = system + user + the `<think>` opener + `environment`
turns; train = reasoning → `</think>` → answer → `<|endoftext|>`).
