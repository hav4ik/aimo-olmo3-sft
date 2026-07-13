#!/usr/bin/env python3
"""
Rollout engine for trace collection.

The single public entry point is :func:`collect_rollout`, which runs one model rollout and returns a
structured trace dictionary. There is ONE conversation loop: every turn, the model may emit a
``<function_calls>`` block, which we self-parse and execute, feeding the result back, until a turn
emits no call (or the deadline/turn budget is hit). The ``tools=`` argument only controls whether the
``<functions>`` schema is rendered in the prompt — it does NOT branch the loop. So "tool" vs "no-tool"
modes differ *purely in the prompt*; a no-schema prompt that still emits a call is honored. The
task-specific prompt construction and answer extraction live in the dispatcher (``run_eval.py``).

Tool-calling design
-------------------
We deliberately do **not** rely on vLLM's ``--tool-call-parser olmo3``. That parser reconstructs the
model's pythonic call by splitting on newlines and re-joining with commas, which corrupts multi-line
``code="..."`` arguments (a SyntaxError on ~8% of geometry-heavy calls, silently dropping them).

Instead the model emits its *trained* format as plain text::

    <think> ... </think><function_calls>stateful_python_code_exec(code="...")</function_calls>

and we parse the call ourselves (``ast`` on the single call, with a regex fallback) and execute it in
the sandbox. The ``<functions>`` schema is still rendered server-side by the baked chat template when
``tools=`` is passed; we use ``tool_choice="none"`` so the request is accepted even on a server started
without ``--enable-auto-tool-choice`` / ``--tool-call-parser``.

Serving requirements
--------------------
* Chat template baked into ``tokenizer_config.json`` (renders the ``<functions>`` schema from ``tools=``).
* A sandbox reachable via :class:`sandbox_client.LocalSandbox` (Docker, default port 6001) for tool runs.
"""

from __future__ import annotations

import ast
import re
import time
from typing import Any, Optional
from uuid import uuid4

FUNCTION_CALLS_OPEN = "<function_calls>"
FUNCTION_CALLS_CLOSE = "</function_calls>"
END_OF_TEXT = "<|endoftext|>"

# Libraries pre-imported into every sandbox session so tool code can use them without boilerplate.
SANDBOX_PREIMPORT = (
    "import math\n"
    "import numpy\n"
    "import sympy\n"
    "import itertools\n"
    "import collections\n"
    "import mpmath\n"
    "mpmath.mp.dps = 64\n"
)


def extract_tool_calls(assistant_text: str) -> list[Optional[str]]:
    """Extract the ``code`` argument from every ``<function_calls>`` block in ``assistant_text``.

    Returns one entry per block: the code string, or ``None`` if the call could not be parsed.
    Robust to multi-line code and nested quotes — we ``ast``-parse the single call (the thing the
    vLLM olmo3 parser fails at) and fall back to a regex if that raises.
    """
    extracted: list[Optional[str]] = []
    block_pattern = re.escape(FUNCTION_CALLS_OPEN) + r"(.*?)" + re.escape(FUNCTION_CALLS_CLOSE)

    for block in re.finditer(block_pattern, assistant_text, re.DOTALL):
        call_source = block.group(1).strip()
        code = _parse_code_with_ast(call_source)
        if code is None:
            code = _parse_code_with_regex(call_source)
        extracted.append(code)
    return extracted


def _parse_code_with_ast(call_source: str) -> Optional[str]:
    """Parse ``stateful_python_code_exec(code=...)`` via the AST and return the ``code`` literal."""
    match = re.search(r"stateful_python_code_exec\s*\(.*\)", call_source, re.DOTALL)
    if not match:
        return None
    try:
        node = ast.parse(match.group(0).strip(), mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(node, ast.Call):
        return None
    for keyword in node.keywords:
        if keyword.arg == "code":
            try:
                return ast.literal_eval(keyword.value)
            except (ValueError, SyntaxError):
                return None
    if node.args:
        try:
            return ast.literal_eval(node.args[0])
        except (ValueError, SyntaxError):
            return None
    return None


def _parse_code_with_regex(call_source: str) -> Optional[str]:
    """Last-resort extraction of a ``code='...'`` / ``code="..."`` argument when the AST parse fails."""
    match = re.search(r"code\s*=\s*(['\"])(.*?)\1\s*\)?\s*$", call_source, re.DOTALL)
    if not match:
        return None
    quote, body = match.group(1), match.group(2)
    try:
        return ast.literal_eval(quote + body + quote)
    except (ValueError, SyntaxError):
        # Best-effort: the body is most often genuine raw code the AST choked on (e.g. real newlines
        # inside the quotes, which is not a valid single-quoted literal but IS runnable). We return it
        # rather than None so that case still executes; the sandbox surfaces a SyntaxError if it was a
        # truly broken/escaped literal. (Rare path: ~0.5% of calls reach the regex fallback.)
        return body


# An exception class line ("SyntaxError: ...", "ValueError: ...") at the start of a line. SyntaxError
# in particular carries no "Traceback" line. Anchored to avoid false positives like "rel_error = 0.5".
_ERROR_LINE = re.compile(r"(?m)^\s*\w*(?:Error|Exception)\s*:")


def _looks_like_error(tool_output: str) -> bool:
    """Heuristic for whether a sandbox result represents a failure."""
    return (
        tool_output.startswith("[ERROR]")
        or "Traceback (most recent call last)" in tool_output
        or "timed out" in tool_output
        or bool(_ERROR_LINE.search(tool_output))
    )


async def _execute_in_sandbox(sandbox, session_id, code, timeout, max_output_chars) -> str:
    """Run ``code`` as an IPython cell in the sandbox and return its combined stdout/stderr."""
    output, _ = await sandbox.execute_code(
        generated_code=code,
        session_id=session_id,
        timeout=timeout,
        max_output_characters=max_output_chars,
    )
    stdout, stderr = output.get("stdout", ""), output.get("stderr", "")
    if stderr:
        return f"{stdout.rstrip()}\n{stderr}" if stdout.strip() else stderr
    return stdout if stdout.strip() else "[WARN] No output. Use print() to see results."


async def collect_rollout(
    *,
    client,
    model: str,
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    sandbox=None,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_tokens: int = 0,
    max_turns: int = 64,
    python_timeout: int = 60,
    max_output_chars: int = 3000,
    deadline: Optional[float] = None,
    extra_body: Optional[dict] = None,
) -> dict[str, Any]:
    """Run a single rollout and return a trace dictionary.

    Parameters
    ----------
    client, model : the OpenAI-compatible client and served model id.
    messages      : the fully built initial chat messages (system + user).
    tools         : tool schema list. Controls ONLY whether the ``<functions>`` schema is rendered in
                    the prompt (``None`` => not rendered). It does NOT change the conversation loop:
                    the engine always self-parses ``<function_calls>`` and executes them, so a no-schema
                    prompt that still emits a tool call is honored. (Difference between "tool" and
                    "no-tool" modes is therefore purely the prompt.)
    sandbox       : a ``LocalSandbox`` instance. Recommended for BOTH modes so emitted calls can run;
                    if ``None``, an emitted call gets an "[ERROR] no sandbox" result and the loop
                    continues. A turn that emits no call ends the rollout (so plain prompts stop fast).
    max_tokens    : per-turn output cap; ``<= 0`` lets the server fill the context window.
    deadline      : optional wall-clock epoch time (``time.time()`` units). When set, generation is
                    aborted as soon as it is exceeded — both between turns and *mid-stream*. The
                    request is streamed, so closing the stream disconnects the HTTP request and vLLM
                    aborts the in-flight decode, freeing the GPU slot. ``None`` => no time limit.

    The returned dict always includes ``finish_reason`` and ``generation`` (concatenated assistant
    text). ``finish_reason == "deadline"`` indicates a time-limit abort. On an unrecoverable error it
    returns ``finish_reason == "error"`` with an ``error`` message rather than raising, so callers can
    record-and-continue.
    """
    # `tools` controls ONLY whether the <functions> schema is rendered into the prompt. The
    # conversation loop is IDENTICAL with or without it: we always self-parse <function_calls> and
    # execute them, so a no-schema ("no-tool") prompt that nonetheless emits a tool call is honored.
    render_functions = tools is not None
    started_at = time.time()
    conversation = [dict(message) for message in messages]
    request_extra_body = {"skip_special_tokens": False}
    if extra_body:
        request_extra_body.update(extra_body)

    sandbox_session_id = uuid4().hex if sandbox is not None else None
    finish_reason = "max_turns"
    total_completion_tokens = 0
    last_prompt_tokens = 0
    num_tool_calls = 0
    num_tool_errors = 0
    turn_log: list[dict] = []

    try:
        if sandbox_session_id is not None:
            try:
                await sandbox.execute_code(
                    generated_code=SANDBOX_PREIMPORT, session_id=sandbox_session_id, timeout=30
                )
            except Exception:
                pass  # pre-import is best-effort; the model's own imports usually suffice.

        # Same loop for both paths: a turn that emits no tool call ends the rollout, so a no-tool
        # prompt naturally stops after one turn while still being allowed to call tools if it wants.
        for turn_index in range(max_turns):
            if deadline is not None and time.time() > deadline:
                finish_reason = "deadline"
                break

            request = {
                "model": model,
                "messages": conversation,
                "temperature": temperature,
                "top_p": top_p,
                # Stream so a deadline can abort the decode: closing the stream disconnects the HTTP
                # request, and vLLM aborts the in-flight generation (freeing the GPU). A blocking
                # call cannot be cancelled once issued.
                "stream": True,
                "stream_options": {"include_usage": True},
                "extra_body": request_extra_body,
            }
            if render_functions:
                # tools= makes the template render the schema; tool_choice="none" avoids requiring the
                # server-side parser (we parse the emitted calls ourselves).
                request["tools"] = tools
                request["tool_choice"] = "none"
            if max_tokens and max_tokens > 0:
                request["max_completion_tokens"] = max_tokens

            assistant_text = ""
            reasoning_text = ""
            turn_finish_reason = None
            aborted = False
            stream = client.chat.completions.create(**request)
            try:
                for chunk in stream:
                    if deadline is not None and time.time() > deadline:
                        aborted = True
                        break
                    usage = getattr(chunk, "usage", None)
                    if usage is not None:
                        total_completion_tokens += usage.completion_tokens or 0
                        last_prompt_tokens = usage.prompt_tokens or last_prompt_tokens
                    if not chunk.choices:
                        continue  # usage-only final chunk carries no choices
                    choice = chunk.choices[0]
                    delta = choice.delta
                    if delta is not None:
                        # Most server configs keep the whole <think>...</think> block in `content`. But
                        # a server started with a reasoning parser splits it into `reasoning_content`;
                        # capture it so the think block is never silently lost.
                        if getattr(delta, "reasoning_content", None):
                            reasoning_text += delta.reasoning_content
                        if delta.content:
                            assistant_text += delta.content
                    if choice.finish_reason:
                        turn_finish_reason = choice.finish_reason
            finally:
                try:
                    stream.close()  # disconnect -> server-side abort if still generating
                except Exception:
                    pass

            # If the server split reasoning out, splice it back inline so </think>-based extraction
            # downstream still works (the model was trained on inline think blocks).
            if reasoning_text and "</think>" not in assistant_text:
                assistant_text = f"<think>{reasoning_text}</think>" + assistant_text

            conversation.append({"role": "assistant", "content": assistant_text})

            if aborted:
                turn_log.append({"turn": turn_index, "tool_calls": 0})
                finish_reason = "deadline"
                break

            model_made_a_call = FUNCTION_CALLS_OPEN in assistant_text
            if not model_made_a_call:
                turn_log.append({"turn": turn_index, "tool_calls": 0})
                # A healthy completion always carries a finish_reason on its last content chunk; its
                # absence means the stream ended abnormally (dropped connection) — don't mislabel that
                # as a clean "stop".
                if turn_finish_reason == "length":
                    finish_reason = "length"
                elif turn_finish_reason is None:
                    finish_reason = "incomplete"
                else:
                    finish_reason = "stop"
                break

            calls = extract_tool_calls(assistant_text)
            for call_index, code in enumerate(calls):
                num_tool_calls += 1
                if sandbox_session_id is None:
                    # No sandbox provided (caller opted out of execution): tell the model so it can
                    # proceed without the result, rather than crashing.
                    result = "[ERROR] No code-execution sandbox is available."
                elif not code:
                    result = "[ERROR] Could not parse code from the function call."
                else:
                    try:
                        result = await _execute_in_sandbox(
                            sandbox, sandbox_session_id, code, python_timeout, max_output_chars
                        )
                    except Exception as exc:
                        result = f"[ERROR] {type(exc).__name__}: {exc}"
                if _looks_like_error(result):
                    num_tool_errors += 1
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": f"call_{turn_index}_{call_index}",
                        "name": "stateful_python_code_exec",
                        "content": result,
                    }
                )
            turn_log.append({"turn": turn_index, "tool_calls": len(calls)})
        else:
            finish_reason = "max_turns"

    except Exception as exc:
        return {
            "finish_reason": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "generation": "".join(
                m["content"] for m in conversation if m.get("role") == "assistant" and m.get("content")
            ),
            "conversation": conversation,
            "num_turns": len(turn_log),
        }
    finally:
        if sandbox_session_id is not None:
            try:
                await sandbox.delete_session(sandbox_session_id)
            except Exception:
                pass
            try:
                await sandbox.close()
            except Exception:
                pass

    generation = "".join(
        m["content"] for m in conversation if m.get("role") == "assistant" and m.get("content")
    )
    return {
        "finish_reason": finish_reason,
        # "truncated" keeps its established meaning (ran out of room/turns), consistent with the
        # downstream analyzers. A "deadline" abort is an operational budget event, not degeneration,
        # so it is queryable via finish_reason but deliberately NOT folded in here.
        "truncated": finish_reason in ("length", "max_turns"),
        "num_turns": len(turn_log),
        "num_tool_calls": num_tool_calls,
        "num_tool_errors": num_tool_errors,
        "num_completion_tokens": total_completion_tokens,
        "num_prompt_tokens": last_prompt_tokens,
        "generation": generation,
        "conversation": conversation,
        "turns": turn_log,
        "generation_time": round(time.time() - started_at, 2),
    }
