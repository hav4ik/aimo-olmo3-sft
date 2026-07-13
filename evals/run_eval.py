#!/usr/bin/env python3
"""
Task dispatcher for trace collection.

A single rollout engine (:func:`traces_engine.collect_rollout`) is driven through six prompt modes::

    proof_notool   / proof_tool      rigorous-proof prompt ("## Solution"); the tool variant is hybrid
    analysis_notool / analysis_tool  0/1/6/7 grading prompt;                the tool variant is hybrid
    boxed_notool   / boxed_tool      \\boxed{} final-answer prompt

Prompt fidelity
---------------
The model is extremely sensitive to prompt wording, so the prompts the data agent handed us are
reproduced **byte-for-byte**:

* ``proof_notool``    == cascade2 ``build_proof_messages(self_verify=False)``
* ``analysis_*``      == cascade2 ``build_analysis_messages`` (prefix / separator / suffix)
* ``boxed_notool``    == ``boxed_cot_prompt.build_messages`` (DOUBLE-brace ``\\boxed{{}}``)
* ``boxed_tool``      == ``tool_boxed_prompt`` (SINGLE-brace ``\\boxed{}``)

The two *tool* proof/analysis modes are our own hybrid: the tool system prompt + schema combined with
the (unchanged) proof/analysis user turn. The byte-exactness of the reproduced prompts is checked at
import time by :func:`_verify_prompt_fidelity` against the source modules in ``boxed_prompts/`` whenever
those modules are importable (it is a no-op otherwise, so the dispatcher still runs standalone).

Tooling
-------
Tool modes pass the ``FUNCTIONS`` schema to the engine; the server's baked chat template renders it into
a ``<functions>`` block and the engine self-parses the emitted ``<function_calls>`` (no reliance on
vLLM's ``--tool-call-parser``). Input JSONL records::

    proof / boxed : {"id", "problem", "answer"?}
    analysis      : {"id", "problem", "solution"}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import httpx
from openai import OpenAI

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from traces_engine import collect_rollout

try:
    from sandbox_client import LocalSandbox
except ImportError:
    sys.path.insert(0, str(THIS_DIR.parent / "src"))
    from solvers.sandbox_client import LocalSandbox

log = logging.getLogger("run_eval")

DEFAULT_MODEL = "chankhavu/smolmo-32b-nvfp4-step17000"
END_OF_TEXT = "<|endoftext|>"
END_OF_TURN = "<|im_end|>"

# Tasks whose prompt requires a candidate solution to grade (vs. just a problem statement).
# Tasks whose prompt requires a candidate solution; refine additionally requires its evaluation.
TASKS_NEEDING_SOLUTION = {"analysis_notool", "analysis_tool", "refine_notool", "refine_tool"}
TASKS_NEEDING_EVALUATION = {"refine_notool", "refine_tool"}
ALL_TASKS = [
    "proof_notool", "proof_tool",
    "analysis_notool", "analysis_tool",
    "boxed_notool", "boxed_tool",
    "refine_notool", "refine_tool",
]

# --------------------------------------------------------------------------------------------------
# Byte-exact prompt literals (see module docstring; verified against boxed_prompts/ at import time).
# --------------------------------------------------------------------------------------------------

# Identical system prompt for the no-tool proof and no-tool boxed modes.
SYSTEM_NO_TOOL = (
    "You are an expert mathematical assistant. Provide rigorous, complete proofs. "
    "You are not allowed to use tools."
)
SYSTEM_BOXED_NO_TOOL = SYSTEM_NO_TOOL

# Tool system prompt: announces the <functions> protocol the chat template renders.
SYSTEM_WITH_TOOL = (
    "You are an expert mathematical assistant. Provide rigorous, complete solutions. "
    "You are provided with function signatures within <functions></functions> XML tags. "
    "You may call one or more functions to assist with the user query. Output any function calls "
    "within <function_calls></function_calls> XML tags. Don't make assumptions about what values to "
    "plug into functions."
)

FUNCTIONS_SCHEMA_JSON = (
    '[{"type": "function", "function": {"name": "stateful_python_code_exec", "description": "Call '
    'this function to execute Python code in a stateful Jupyter notebook environment. Python will '
    'respond with the output of the execution or time out after 120.0 seconds.", "parameters": '
    '{"type": "object", "properties": {"code": {"type": "string", "description": "Code to execute"}}, '
    '"required": ["code"]}}}]'
)
FUNCTIONS_SCHEMA = json.loads(FUNCTIONS_SCHEMA_JSON)

PROOF_USER_PREFIX = (
    "Your task is to solve a given problem. The problem may ask you to prove a statement, or ask for "
    "an answer. If finding an answer is required, you should come up with the answer, and your final "
    "solution should also be a rigorous proof of that answer being valid.\n\nYour final solution to "
    "the problem should be exceptionally comprehensive and easy-to-follow, which will be rated "
    "according to the following evaluation instruction:\n\n```txt\nHere is the instruction to "
    "evaluate the quality of a solution to a problem. The problem may ask for a proof of statement, "
    "or ask for an answer. If finding an answer is required, the solution should present the answer, "
    "and it should also be a rigorous proof of that answer being valid.\n\nPlease evaluate the "
    "solution and score it according to the following criteria:\n- If the solution is completely "
    "correct, with all steps executed properly and clearly demonstrated, then the score is 1\n- If "
    "the solution is generally correct, but with some details omitted or minor errors, then the "
    "score is 0.5\n- If the solution does not actually address the required problem, contains fatal "
    "errors, or has severe omissions, then the score is 0\n\nAdditionally, referencing anything from "
    "any paper does not save the need to prove the reference. It's okay IF AND ONLY IF the solution "
    "also presents a valid proof of the reference argument(s); otherwise, if the solution omits the "
    "proof or if the proof provided is not completely correct, the solution should be scored "
    "according to the criteria above, and definitely not with a score of 1\n```\n\nYour final "
    "response should be in the following format:\n\n## Solution // Your final solution should start "
    "with this exact same markdown title\n... // Your final solution to the problem here. You should "
    "try your best to optimize the quality of your solution according to the evaluation instruction "
    "above before finalizing it here.\n\n---\n\nHere is your task input:\n\n## Problem\n"
)

ANALYSIS_USER_PREFIX = (
    "Carefully analyze the given problem statement and the proposed solution, and then write out "
    "your analysis regarding the correctness of the proposed solution. \n\nAfter the analysis, you "
    "must provide a score based on the following grading scale:\n\n- 0: Incorrect - The solution is "
    "completely incorrect or irrelevant.\n- 1: Partial - The solution is partially correct but has "
    "significant errors or omissions.\n- 6: Almost - The solution is almost correct but contains "
    "minor errors or inaccuracies.\n- 7: Correct - The solution is fully correct and complete.\n\n\n"
    "Problem:\n"
)
ANALYSIS_SOLUTION_SEPARATOR = "\n\nSolution:\n"
ANALYSIS_USER_SUFFIX = (
    "Analyze the solution carefully, then provide your grade as a single number (0, 1, 6, or 7) in "
    "\\boxed{{}}."
)

# DOUBLE-brace in the no-tool boxed prompt, SINGLE-brace in the tool boxed prompt — intentional and
# preserved exactly as the data agent delivered them.
BOXED_NO_TOOL_PREFIX = "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n"
BOXED_WITH_TOOL_PREFIX = "Please reason step by step, and put your final answer within \\boxed{}.\n\n"

# Refine task (NEW; not in training). The refine USER turn is the proof prompt VERBATIM
# (``PROOF_USER_PREFIX`` + problem) with two extra input sections and a closing instruction appended.
# Only these three glue strings are new; the reused proof framing/rubric/format stays byte-exact.
REFINE_PREV_SOLUTION_SEP = "\n\n## Previous Solution\n"
REFINE_EVALUATION_SEP = "\n\n## Evaluation\n"
REFINE_SUFFIX = (
    "\n\nThe ## Previous Solution above is an earlier attempt at this problem, and ## Evaluation is "
    "an assessment of that attempt's quality. Produce an improved solution: address every error, gap, "
    "and omission identified in the evaluation, and otherwise strengthen the rigor, completeness, and "
    "clarity of the argument. Present your improved final solution using the ## Solution format "
    "described above."
)


def build_proof_user(problem: str) -> str:
    return PROOF_USER_PREFIX + problem


def build_refine_user(problem: str, solution: str, evaluation: str) -> str:
    return (PROOF_USER_PREFIX + problem + REFINE_PREV_SOLUTION_SEP + solution
            + REFINE_EVALUATION_SEP + evaluation + REFINE_SUFFIX)


def build_analysis_user(problem: str, solution: str) -> str:
    return ANALYSIS_USER_PREFIX + problem + ANALYSIS_SOLUTION_SEPARATOR + solution + "\n\n" + ANALYSIS_USER_SUFFIX


def build_boxed_no_tool_user(problem: str) -> str:
    return BOXED_NO_TOOL_PREFIX + problem


def build_boxed_with_tool_user(problem: str) -> str:
    return BOXED_WITH_TOOL_PREFIX + problem


def build_messages(
    task: str, problem: str, solution: str = "", evaluation: str = ""
) -> tuple[list[dict], Optional[list[dict]]]:
    """Return ``(messages, tools)`` for ``task``.

    ``tools`` is ``None`` for no-tool modes (plain single-turn rollout) and the parsed ``FUNCTIONS``
    schema for tool modes (multi-turn self-parsed tool loop). ``solution`` is required by analysis and
    refine tasks; ``evaluation`` additionally by refine. Raises :class:`ValueError` on an unknown task.
    """
    if task == "proof_notool":
        return _chat(SYSTEM_NO_TOOL, build_proof_user(problem)), None
    if task == "proof_tool":
        return _chat(SYSTEM_WITH_TOOL, build_proof_user(problem)), FUNCTIONS_SCHEMA
    if task == "analysis_notool":
        return _chat(SYSTEM_NO_TOOL, build_analysis_user(problem, solution)), None
    if task == "analysis_tool":
        return _chat(SYSTEM_WITH_TOOL, build_analysis_user(problem, solution)), FUNCTIONS_SCHEMA
    if task == "boxed_notool":
        return _chat(SYSTEM_BOXED_NO_TOOL, build_boxed_no_tool_user(problem)), None
    if task == "boxed_tool":
        return _chat(SYSTEM_WITH_TOOL, build_boxed_with_tool_user(problem)), FUNCTIONS_SCHEMA
    if task == "refine_notool":
        return _chat(SYSTEM_NO_TOOL, build_refine_user(problem, solution, evaluation)), None
    if task == "refine_tool":
        return _chat(SYSTEM_WITH_TOOL, build_refine_user(problem, solution, evaluation)), FUNCTIONS_SCHEMA
    raise ValueError(f"unknown task: {task!r}")


def _chat(system_content: str, user_content: str) -> list[dict]:
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


# --------------------------------------------------------------------------------------------------
# Answer extraction
# --------------------------------------------------------------------------------------------------

_VALID_ANALYSIS_SCORES = {0, 1, 6, 7}
_THINK_CLOSE = "</think>"


def _iter_boxed(text: str):
    """Yield the brace-BALANCED contents of each ``\\boxed{...}`` (handles nesting like
    ``\\boxed{\\frac{1}{2}}``). A naive ``\\boxed\\{([^}]*)\\}`` regex would truncate at the first
    ``}``. An unbalanced (truncated) box is skipped."""
    needle = "\\boxed{"
    i = 0
    while True:
        j = text.find(needle, i)
        if j < 0:
            return
        k = j + len(needle)
        depth, start = 1, k
        while k < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            k += 1
        if depth == 0:
            yield text[start:k - 1]
        i = k


def _last_boxed(text: str) -> Optional[str]:
    boxes = [b.strip() for b in _iter_boxed(text)]
    return boxes[-1] if boxes else None


def extract_result(task: str, generation: str) -> dict[str, Any]:
    """Pull the task-specific field out of the full generation text."""
    if task.startswith("proof") or task.startswith("refine"):
        return {"solution": _extract_proof_solution(generation)}
    if task.startswith("analysis"):
        return {"score": _extract_analysis_score(generation)}
    return {"boxed": _last_boxed(generation)}


def _extract_proof_solution(generation: str) -> str:
    """Return everything from the final ``## Solution`` heading onward, stripped of stop tokens.

    Searches only the post-``</think>`` final-answer region so a ``## Solution`` mentioned *inside*
    the reasoning of a degenerate/truncated rollout is not mistaken for the answer. (If the rollout
    never closed ``</think>``, the whole text is searched as a fallback.)
    """
    tail = generation.rsplit(_THINK_CLOSE, 1)[-1]
    start = tail.rfind("## Solution")
    if start < 0:
        return ""
    return tail[start:].replace(END_OF_TEXT, "").replace(END_OF_TURN, "").strip()


def _extract_analysis_score(generation: str) -> Optional[int]:
    """Return the final boxed grade BUCKETED to the nearest valid 0/1/6/7 value.

    Any numeric boxed value is accepted (floats are fine) and snapped to the closest grade (ties go to
    the lower grade, e.g. 6.5 -> 6). Returns ``None`` only when there is no numeric boxed value at all.
    """
    raw = _last_boxed(generation)
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return min(_VALID_ANALYSIS_SCORES, key=lambda g: (abs(g - value), g))


# --------------------------------------------------------------------------------------------------
# Output (one append-only JSONL file per problem id, resume-aware)
# --------------------------------------------------------------------------------------------------

_write_lock = threading.Lock()


def append_record(output_dir: Path, record: dict) -> None:
    """Append one rollout record to ``<output_dir>/<id>.jsonl`` under a process-wide lock."""
    path = output_dir / f"{record['id']}.jsonl"
    with _write_lock:
        with open(path, "a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def completed_sessions(output_dir: Path, problem_id: str) -> set[int]:
    """Return the set of ``session_index`` values already recorded for ``problem_id`` (for --resume)."""
    path = output_dir / f"{problem_id}.jsonl"
    if not path.exists():
        return set()
    done: set[int] = set()
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["session_index"])
            except (json.JSONDecodeError, KeyError):
                continue  # tolerate a partially written trailing line
    return done


def read_problem_text(record: dict) -> str:
    """Extract the problem statement from an input record, tolerating a few common schemas."""
    if record.get("problem"):
        return record["problem"]
    for message in reversed(record.get("messages", [])):
        if message.get("role") == "user":
            return message["content"]
    return record.get("text", "")


# --------------------------------------------------------------------------------------------------
# Rollout
# --------------------------------------------------------------------------------------------------


def run_single_rollout(
    args: argparse.Namespace, client: OpenAI, problem: str, solution: str, evaluation: str = ""
) -> dict[str, Any]:
    """Build the prompt for ``args.task``, run one rollout, and attach the extracted answer."""
    messages, tools = build_messages(args.task, problem, solution, evaluation)
    # Always provide a sandbox: the loop is identical for tool/no-tool, so a no-tool prompt may still
    # emit a tool call and we let it execute. `tools` only controls whether the schema is rendered.
    sandbox = LocalSandbox(host=args.sandbox_host, port=args.sandbox_port)
    trace = asyncio.run(
        collect_rollout(
            client=client,
            model=args.model_name,
            messages=messages,
            tools=tools,
            sandbox=sandbox,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            max_turns=args.max_turns,
            python_timeout=args.python_timeout,
            max_output_chars=args.max_output_characters,
        )
    )
    trace.update(extract_result(args.task, trace.get("generation", "")))
    return trace


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace dispatcher: six prompt modes, one rollout engine.")
    parser.add_argument("--task", required=True, choices=ALL_TASKS)
    parser.add_argument("--input", required=True, help="input JSONL of problems (and solutions, for analysis)")
    parser.add_argument("--output_dir", required=True, help="directory for per-problem JSONL traces")
    parser.add_argument("--n_sessions", type=int, default=4, help="rollouts per problem")
    parser.add_argument("--max_parallel", type=int, default=16, help="concurrent rollouts")
    parser.add_argument("--server_addr", required=True, help="host:port or full URL of the OpenAI-compatible server")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--sandbox_host", default="127.0.0.1")
    parser.add_argument("--sandbox_port", type=int, default=6001)
    # ⚠️ KEEP temperature at 1.0 (the production setting). This olmo checkpoint family is UNSTABLE at
    # lower temperatures (0.6 and below) — it degenerates into repetitive/looping rollouts that truncate
    # and tank accuracy. 1.0 stabilizes it. Do NOT lower it "to reduce variance" for Avg@k — that
    # backfires here. (temp=0 is the worst: ~6x more loops.) top_p stays 0.95.
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="KEEP AT 1.0 — this model degenerates (repetitive loops) below ~1.0; do not lower")
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=0, help="per-turn output cap; <=0 fills the context window")
    parser.add_argument("--max_turns", type=int, default=64, help="max tool turns (tool modes only)")
    parser.add_argument("--python_timeout", type=int, default=60, help="per-cell sandbox timeout (seconds)")
    parser.add_argument("--max_output_characters", type=int, default=3000, help="sandbox output truncation length")
    parser.add_argument("--resume", action="store_true", help="skip problem/session pairs already recorded")
    return parser.parse_args(argv)


def build_client(server_addr: str, api_key: Optional[str], read_timeout: float = 300.0) -> OpenAI:
    base_url = server_addr if server_addr.startswith("http") else f"http://{server_addr}"
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    key = api_key or os.environ.get("OPENAI_API_KEY", "sk-local")
    # A FINITE read timeout is essential: the engine streams and checks the wall-clock deadline only
    # when a chunk arrives, so a server that stalls mid-stream (socket open, no tokens) would
    # otherwise block a worker thread forever (deadline never re-evaluated -> process hangs at exit).
    #
    # NOTE: with streaming, `read` is the max SILENCE BETWEEN TOKENS, not the total request time, and
    # it resets on every chunk. So it is independent of the per-rollout phase `deadline` (which may be
    # many minutes, or disabled) — `read` only needs to exceed the worst legitimate gap:
    # time-to-first-token (prompt prefill, largest for long grading prompts under load) and inter-token
    # gaps under heavy concurrency. 300s gives generous TTFT headroom while still catching a true stall.
    # HTTP-level timeouts are ALWAYS kept finite: they are the safety net the streaming-abort design
    # relies on (an infinite read can hang a worker forever on a stalled stream). A non-positive value
    # therefore falls back to the safe default rather than disabling.
    read = read_timeout if (read_timeout and read_timeout > 0) else 300.0
    timeout = httpx.Timeout(connect=10.0, read=read, write=60.0, pool=read)
    return OpenAI(base_url=base_url, api_key=key, timeout=timeout)


def plan_work(args: argparse.Namespace, problems: list[dict]) -> list[dict[str, Any]]:
    """Expand the input problems into the list of (problem, session) rollouts still to run."""
    output_dir = Path(args.output_dir)
    # Duplicate ids would collide on the per-id output file AND mask each other under --resume
    # (the second problem's sessions look already-done). Fail loudly rather than corrupt/lose data.
    ids = [str(r["id"]) for r in problems if "id" in r]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate problem ids in input (would collide on output/resume): {dupes}")

    work: list[dict[str, Any]] = []
    for record in problems:
        if "id" not in record:
            log.warning("input record missing 'id'; skipping: %.80s", json.dumps(record, ensure_ascii=False))
            continue
        problem_id = str(record["id"])
        problem_text = read_problem_text(record)
        if not problem_text:
            log.warning("%s: no problem text found; skipping", problem_id)
            continue
        solution = record.get("solution") or record.get("sol") or ""
        evaluation = record.get("evaluation") or record.get("eval") or ""
        if args.task in TASKS_NEEDING_SOLUTION and not solution:
            log.warning("%s: %s requires a solution but none was found; skipping", problem_id, args.task)
            continue
        if args.task in TASKS_NEEDING_EVALUATION and not evaluation:
            log.warning("%s: %s requires an evaluation but none was found; skipping", problem_id, args.task)
            continue
        already_done = completed_sessions(output_dir, problem_id) if args.resume else set()
        expected_answer = record.get("answer", record.get("expected_answer"))
        for session_index in range(args.n_sessions):
            if session_index in already_done:
                continue
            work.append(
                {
                    "id": problem_id,
                    "problem": problem_text,
                    "solution": solution,
                    "evaluation": evaluation,
                    "session_index": session_index,
                    "expected_answer": expected_answer,
                }
            )
    return work


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    client = build_client(args.server_addr, args.api_key)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input) as handle:
        problems = [json.loads(line) for line in handle if line.strip()]
    log.info("task=%s | %d problems x %d sessions -> %s", args.task, len(problems), args.n_sessions, output_dir)

    work = plan_work(args, problems)
    if not work:
        log.info("nothing to do (all sessions already complete?)")
        return 0
    log.info("running %d rollouts, max_parallel=%d", len(work), args.max_parallel)

    def execute(item: dict[str, Any]) -> dict[str, Any]:
        trace = run_single_rollout(args, client, item["problem"], item["solution"], item["evaluation"])
        trace["id"] = item["id"]
        trace["session_index"] = item["session_index"]
        trace["expected_answer"] = item["expected_answer"]
        trace["task"] = args.task
        trace["problem"] = item["problem"]
        if args.task in TASKS_NEEDING_SOLUTION:
            trace["graded_solution"] = item["solution"]
        if args.task in TASKS_NEEDING_EVALUATION:
            trace["graded_evaluation"] = item["evaluation"]
        return trace

    succeeded = failed = 0
    total = len(work)
    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        future_to_item = {pool.submit(execute, item): item for item in work}
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            label = f"{item['id']}:{item['session_index']}"
            try:
                trace = future.result()
            except Exception as exc:  # defensive: a worker should not normally raise
                failed += 1
                log.error("  %s failed: %s: %s", label, type(exc).__name__, exc)
                continue
            if trace.get("finish_reason") == "error":
                failed += 1
                log.error("  %s engine error: %s", label, trace.get("error"))
                continue
            append_record(output_dir, trace)
            succeeded += 1
            summary = trace.get("score", trace.get("boxed", bool(trace.get("solution"))))
            print(
                f"  [{succeeded}/{total}] {label} {args.task} "
                f"finish={trace['finish_reason']} tools={trace['num_tool_calls']} out={summary}",
                flush=True,
            )

    log.info("Done. %d ok, %d errors -> %s", succeeded, failed, output_dir)
    return 0


# --------------------------------------------------------------------------------------------------
# Prompt-fidelity self-check (runs at import; silent no-op if the source modules are unavailable)
# --------------------------------------------------------------------------------------------------


def _verify_prompt_fidelity() -> None:
    """Assert the baked prompts match the boxed_prompts/ source modules, byte for byte.

    This guards the project's hard constraint that the reproduced prompts are identical to what the
    data agent delivered. It is skipped silently when those modules cannot be imported so the
    dispatcher remains runnable on its own.
    """
    sys.path.insert(0, str(THIS_DIR / "boxed_prompts"))
    try:
        import cascade2_prompts as cascade2
        import boxed_cot_prompt
        import tool_boxed_prompt
    except ImportError:
        return
    try:
        import refine_prompt
    except ImportError:
        refine_prompt = None

    sample_problem = "PROBE-PROBLEM"
    sample_solution = "PROBE-SOLUTION"
    sample_evaluation = "PROBE-EVALUATION"

    checks: list[tuple[str, bool]] = []

    if hasattr(cascade2, "build_proof_messages"):
        reference = cascade2.build_proof_messages(sample_problem, self_verify=False)
        ours, _ = build_messages("proof_notool", sample_problem)
        checks.append(("proof_notool", ours == reference))

    if hasattr(cascade2, "build_analysis_messages"):
        reference = cascade2.build_analysis_messages(sample_problem, sample_solution)
        ours, _ = build_messages("analysis_notool", sample_problem, sample_solution)
        checks.append(("analysis_notool", ours == reference))

    if hasattr(boxed_cot_prompt, "build_messages"):
        reference = boxed_cot_prompt.build_messages(sample_problem)
        ours, _ = build_messages("boxed_notool", sample_problem)
        checks.append(("boxed_notool", ours == reference))

    if hasattr(tool_boxed_prompt, "build_user_content") and hasattr(tool_boxed_prompt, "SYSTEM_CONTENT"):
        ours, _ = build_messages("boxed_tool", sample_problem)
        reference = [
            {"role": "system", "content": tool_boxed_prompt.SYSTEM_CONTENT},
            {"role": "user", "content": tool_boxed_prompt.build_user_content(sample_problem)},
        ]
        checks.append(("boxed_tool", ours == reference))

    if refine_prompt is not None and hasattr(refine_prompt, "build_refine_messages"):
        for task, use_tools in (("refine_notool", False), ("refine_tool", True)):
            reference = refine_prompt.build_refine_messages(
                sample_problem, sample_solution, sample_evaluation, use_tools=use_tools
            )
            ours, _ = build_messages(task, sample_problem, sample_solution, sample_evaluation)
            checks.append((task, ours == reference))

    mismatched = [name for name, ok in checks if not ok]
    if mismatched:
        raise AssertionError(
            "prompt fidelity check FAILED for: "
            + ", ".join(mismatched)
            + " — the baked literals have drifted from boxed_prompts/."
        )


_verify_prompt_fidelity()


if __name__ == "__main__":
    raise SystemExit(main())
