# proof-redesign — source tables & provenance

Workspace for the **re-designed** proof dataset (v2). Build everything under `/mnt/data`.

## Upstream source
- **`nvidia/Nemotron-Cascade-2-SFT-Data`**, `math/` subset, downloaded to
  `/mnt/data/Nemotron-Cascade-2-SFT-Data/math/`:
  - `math_proof.parquet` — **816,585 rows** (proof problems: solve / rubric-evaluate / grade).
  - `math_notool.parquet` — 2,142,332 rows (numeric-answer "\boxed{}" problems). *Not used yet.*
  - `math_tool.parquet` — **2,267,447 rows** (numeric-answer math solved with a stateful Python
    interpreter — tool-augmented, NOT proofs; also some judge/grader rows). Downloaded Jun 2026
    (70.33 GB jsonl → parquet). Converted to **Olmo-3 native tool format** →
    `tables/math_tool_olmo.parquet` (+ `math_tool_quarantine.parquet`). See **`CASCADE_TO_OLMO.md`**
    for the converter, the Olmo-3 tool-format spec, the validation gate, the adversarial-audit
    history, and the vLLM serving recipe (`--tool-call-parser olmo3 --reasoning-parser olmo3`).
- Original `*.jsonl` were **deleted** after a verified lossless jsonl→parquet conversion
  (row counts matched). The parquets are the complete source.
- `math_proof.parquet` carries the original Cascade fields (`domain`, `source`, `messages`,
  `generator`) **plus** materialized `task_type`, `problem_id`, `num_tokens`
  (added by `aimo-proof-pilot/scripts/materialize_labels.py`). `problem_id` =
  sha256 of the aggressive-normalized bare problem statement.

## Tables in this workspace (`tables/`)

Both are direct splits of `math_proof.parquet` by `task_type`. Every row keeps its source
provenance via the `source` (upstream dataset) and `generator` (DeepSeek model) columns.

### `tables/solutions.parquet` — 417,454 rows
- `task_type == "solution"`. Columns: `domain, source, messages, generator, task_type, problem_id, num_tokens`.
- The assistant writes a proof. **~90.8% are the self-evaluating format** (`## Solution` →
  `## Self Evaluation` → self-graded `\boxed{0/0.5/1}`); ~9.2% plain (see
  `aimo-proof-pilot/SOLUTION_PROMPT_FORMATS.md`). The bare `## Solution` proof and the
  self-eval score are **not yet parsed out** — to be added.

### `tables/assessment.parquet` — 399,131 rows
- `task_type ∈ {"evaluation","analysis"}` (the **type marker**). Same 7 columns **plus**:
  - **`score`** (float64) — raw, parsed from the **last `\boxed{...}`** in the assistant turn.
    - `evaluation` (rubric, 201,121 rows): ∈ {0, 0.5, 1}.
    - `analysis` (grading, 198,010 rows): ∈ {0, 1, 6, 7}.
  - **`score_normalized`** (float64) — pooled 3-bucket scale (`0.0` fail / `0.5` almost / `1.0` correct):
    - analysis: `0→0.0, 1→0.0, 6→0.5, 7→1.0`
    - evaluation: `0→0.0, 0.5→0.5, 1→1.0`
- **0 nulls** — every assessment row had a parseable score.
- Distributions: raw evaluation `{0:55300, 0.5:30154, 1:115667}`, raw analysis
  `{0:5369, 1:44636, 6:16538, 7:131467}`; normalized `{0.0:105305, 0.5:46692, 1.0:247134}`.

## Build (reproducible)
- `solutions` / split: stream `math_proof.parquet` row-groups, filter by `task_type`, write zstd parquet.
- `assessment`: from the eval+analysis split, parse last `\boxed{}` → `score`, map → `score_normalized`,
  concatenate. (See the inline scripts in the session / to be committed under `aimo-proof-pilot/scripts`.)

## Next (planned, not done)
- Parse the bare `## Solution` proof on both sides → `solution_id = sha256(normalized proof)`;
  link `assessment` rows to the actual `solutions` row they grade (~58% match, per-proof grades).
- Harvest the self-eval `\boxed` score from `solutions` rows as a third per-proof signal.
- Decide math_notool's role.
