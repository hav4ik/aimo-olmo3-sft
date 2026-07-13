# Evaluation — AIME'25 / AIME'26 / HMMT, no-tool and tool

Benchmark harness for the Olmo-3 32B SFT checkpoints on competition math. Every problem is
run in two modes:

- **no-tool** — pure chain-of-thought, the model just reasons and boxes an answer.
- **tool** — the model may call a Python sandbox (`stateful_python_code_exec`) to offload
  computation, then boxes an answer.

Metrics are **Avg@k** (mean per-sample accuracy) and **maj@k** (equivalence-clustered majority
vote), graded with HuggingFace [`math_verify`](https://github.com/huggingface/Math-Verify) — the
same verifier MathArena uses.

The numbers this harness produced are in [`RESULTS_0207.md`](RESULTS_0207.md) (technical) and
[`blog_aime_hmmt.md`](blog_aime_hmmt.md) (narrative).

---

## TL;DR

```bash
pip install -r requirements.txt

# 1. point at your OpenAI-compatible endpoint (vLLM / SGLang serving the checkpoint)
export OLMO_ADDR=host:port
export OLMO_API_KEY=your_api_key
export OLMO_MODEL=chankhavu/smolmo-32b-nvfp4-step17000

# 2. (tool mode only) start a NeMo-Skills sandbox and confirm it answers
python3 check_sandbox.py 127.0.0.1:6001

# 3. run all 3 benchmarks × {no-tool, tool}; writes traces/<bench>_<mode>/<id>.jsonl
./run_bench.sh

# 4. grade a cell
python3 grade_bench.py traces/aime25_notool
```

---

## What's here

| File | Role |
|---|---|
| `run_eval.py`          | **Harness / dispatcher.** Tasks `boxed_notool` and `boxed_tool` build the byte-exact prompts and drive one rollout engine. Also `proof_*`, `analysis_*`, `refine_*` for other workflows. |
| `traces_engine.py`     | `collect_rollout()` — streaming rollout, self-parsed `<function_calls>` tool loop, wall-clock abort. |
| `sandbox_client.py`    | `LocalSandbox` — client for the NeMo-Skills Python sandbox (tool mode only). |
| `check_sandbox.py`     | Pre-flight: confirm the sandbox is up before a tool run. |
| `boxed_prompts/`       | Canonical prompt sources. `run_eval.py` bakes byte-exact copies and self-checks them against these at import. |
| `grade_bench.py`       | **Primary grader.** `math_verify` first, sympy/string fallback. Prints Avg@k / maj@k / truncation. |
| `grade_bench_mv.py`    | Standalone `math_verify`-only re-grader (handy for HMMT symbolic answers). |
| `compare_models.py`    | Per-problem head-to-head + truncation across two trace dirs. |
| `build_audit_dump.py`  | Dump `(pred, gold, verdict)` per session for manual/agent audit of grades. |
| `bench_aime25.jsonl`   | AIME 2025 — 30 problems, `{id, problem, answer}`. |
| `bench_aime26.jsonl`   | AIME 2026 (MathArena `aime_2026`) — 30 problems. |
| `bench_hmmt_feb2025.jsonl` | HMMT February 2025 (MathArena `hmmt_feb_2025`) — 30 problems, ~half symbolic answers. |
| `run_bench.sh`         | Parameterized entry-point runner (env-var driven, no secrets). |
| `ENDPOINTS.example.yaml` | Template for recording your own serving endpoints (copy → `ENDPOINTS.yaml`, git-ignored). |
| `scripts/`             | The **exact** (sanitized) scripts behind the `0207` results, per checkpoint. |

Benchmark input schema is one JSON object per line:

```json
{"id": "aime25-0", "problem": "Find the sum of all integer bases ...", "answer": "70"}
```

---

## Prerequisites

**1. A served checkpoint.** Any OpenAI-compatible endpoint (vLLM, SGLang, …) exposing
`/v1/chat/completions` and `/v1/models`. The harness talks to it via `--server_addr` / `--api_key`
/ `--model_name` (or the `OLMO_*` env vars). The server's chat template must render the
`<functions>` block for tool mode — the engine self-parses the emitted `<function_calls>`, so no
vLLM `--tool-call-parser` is needed.

**2. A Python sandbox (tool mode only).** `sandbox_client.py` targets a
[NeMo-Skills](https://github.com/NVIDIA/NeMo-Skills) sandbox HTTP server. Point it at yours with
`--sandbox_port` (default 6001) or the `NEMO_SKILLS_SANDBOX_HOST` / `NEMO_SKILLS_SANDBOX_PORT`
env vars, and verify connectivity:

```bash
python3 check_sandbox.py 127.0.0.1:6001
```

No-tool runs don't need the sandbox.

---

## The two modes

Both go through `run_eval.py`; only the `--task` differs.

**No-tool** (`boxed_notool`) — single-turn CoT:
```bash
python3 run_eval.py --task boxed_notool --input bench_aime25.jsonl \
  --output_dir traces/aime25_notool --n_sessions 8 --max_parallel 48 \
  --server_addr "$OLMO_ADDR" --api_key "$OLMO_API_KEY" --model_name "$OLMO_MODEL" \
  --temperature 1.0 --top_p 0.95 --max_tokens 0 --resume
```

**Tool** (`boxed_tool`) — multi-turn Python loop:
```bash
python3 run_eval.py --task boxed_tool --input bench_aime25.jsonl \
  --output_dir traces/aime25_tool --n_sessions 8 --max_parallel 24 \
  --max_turns 64 --python_timeout 60 --max_output_characters 600 \
  --sandbox_port 6001 \
  --server_addr "$OLMO_ADDR" --api_key "$OLMO_API_KEY" --model_name "$OLMO_MODEL" \
  --temperature 1.0 --top_p 0.95 --max_tokens 0 --resume
```

Output is one append-only JSONL per problem id under `--output_dir`, one line per session
(`boxed`, `expected_answer`, `finish_reason`, `num_tool_calls`, `generation`, …).
`--resume` skips sessions already recorded, so runs are safely restartable.

---

## Grading

```bash
python3 grade_bench.py traces/aime25_tool            # Avg@k / maj@k / truncation, per problem
python3 compare_models.py --names A,B traces/dirA traces/dirB   # head-to-head + truncation
```

`grade_bench.py` uses `math_verify` as the primary equivalence check and only falls back to
sympy/string normalization when it can't parse a side.

---

## Gotchas that cost us real points

1. **Temperature must be 1.0.** These 32B checkpoints degenerate into repetitive loops below
   ~1.0 (0.6 truncates and tanks accuracy; temp 0 is worst). `run_eval.py` defaults to 1.0 and
   warns loudly — do **not** lower it "to reduce variance."

2. **Grade symbolic answers with `math_verify`, not string match.** AIME answers are integers
   0–999, but ~half of HMMT's are LaTeX (fractions, radicals, π, factorials, multi-value sets).
   A string grader silently under-counts them — and penalizes **tool mode most**, because Python
   emits `9/√23` where the textbook form is `9√23/23`. Switching to `math_verify` moved one HMMT
   tool cell **+22 points** (67 → 90). AIME integer scores are identical under both graders.

3. **Truncation is a real failure mode.** In no-tool mode 8–13% of samples run past the 65k
   context and get cut off mid-reasoning (`finish_reason: length`). Tools cut that to ~0.4–1.2%
   by offloading computation — that's where most of the tool gain comes from. `grade_bench.py`
   reports the truncation rate per cell.

---

## Reproducing the `0207` results

`scripts/` holds the exact runners behind [`RESULTS_0207.md`](RESULTS_0207.md), one per
checkpoint (`proofpilot`, `step17000`, `step20500`), sanitized to read `OLMO_ADDR` / `OLMO_API_KEY`
from the environment. They write to `traces/bench_0207full_<model>_<bench>_<mode>/`.

```bash
export OLMO_ADDR=host:port OLMO_API_KEY=your_key
bash scripts/run_bench_0207full_step17000.sh        # aime25; *_ext.sh does aime26 + hmmt
```

> `step20500`'s serving endpoint crashed mid-run, so only its AIME'25 cells completed —
> see the note in `RESULTS_0207.md`.
