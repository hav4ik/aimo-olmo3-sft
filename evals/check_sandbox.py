#!/usr/bin/env python3
"""Standalone sanity check for the code-execution sandbox used by kaggle_solver.py.

Exercises the SAME path the solver uses (sandbox_client.LocalSandbox.execute_code) and checks:
  1. basic execution + stdout capture
  2. stateful session (ipython keeps variables across calls)
  3. error/traceback is returned (not swallowed)
  4. the `timeout` argument is honored
  5. output is truncated to max_output_characters
Then cleans up the session and closes the client.

Usage:
    python3 check_sandbox.py                 # defaults to 127.0.0.1:6001
    python3 check_sandbox.py 127.0.0.1:6000  # or pass host:port
Exit code 0 = all checks passed, 1 = something failed.
"""
import asyncio
import sys
import uuid

from sandbox_client import LocalSandbox


def _stdout(out: dict) -> str:
    return (out.get("stdout") or "")


def _stderr(out: dict) -> str:
    return (out.get("stderr") or "")


async def main(host: str, port: str) -> int:
    print(f"[*] sandbox target: http://{host}:{port}")
    sb = LocalSandbox(host=host, port=port)
    sid = str(uuid.uuid4())
    passed, failed = 0, 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if ok:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            failed += 1
            print(f"  [FAIL] {name}  {detail}")

    try:
        # 1. basic execution + stdout
        try:
            out, _ = await sb.execute_code(
                generated_code="print(6 * 7)", session_id=sid, timeout=15, max_output_characters=1000
            )
            check("1. basic execution / stdout", _stdout(out).strip() == "42",
                  f"got stdout={_stdout(out)!r} stderr={_stderr(out)!r}")
        except Exception as exc:
            check("1. basic execution / stdout", False, f"raised {type(exc).__name__}: {exc} "
                  "(server unreachable? wrong port? sandbox not running?)")
            print("\n[!] could not execute at all — aborting further checks.")
            return 1

        # 2. stateful session: define a var, then use it in a later call (ipython keeps state)
        await sb.execute_code(generated_code="x = 123", session_id=sid, timeout=15, max_output_characters=1000)
        out, _ = await sb.execute_code(
            generated_code="print(x + 1)", session_id=sid, timeout=15, max_output_characters=1000
        )
        check("2. stateful session (var persists)", _stdout(out).strip() == "124",
              f"got stdout={_stdout(out)!r} (state not preserved across calls?)")

        # 3. error/traceback surfaces
        out, _ = await sb.execute_code(
            generated_code="1/0", session_id=sid, timeout=15, max_output_characters=2000
        )
        combined = _stdout(out) + _stderr(out)
        check("3. error/traceback returned", "ZeroDivisionError" in combined,
              f"got {combined!r}")

        # 4. timeout is honored: sleep longer than the timeout -> should time out, not hang forever
        import time as _t
        t0 = _t.time()
        out, _ = await sb.execute_code(
            generated_code="import time; time.sleep(10)", session_id=sid, timeout=3, max_output_characters=1000
        )
        elapsed = _t.time() - t0
        timed_out = ("timeout" in str(out.get("process_status", "")).lower()
                     or "timed out" in _stderr(out).lower() or elapsed < 8)
        check("4. timeout honored (~3s cap on a 10s sleep)", timed_out,
              f"elapsed={elapsed:.1f}s status={out.get('process_status')!r} stderr={_stderr(out)!r}")

        # 5. output truncation to max_output_characters
        out, _ = await sb.execute_code(
            generated_code="print('A' * 5000)", session_id=sid, timeout=15, max_output_characters=200
        )
        check("5. output truncated to max_output_characters", len(_stdout(out)) <= 600,
              f"got {len(_stdout(out))} chars (cap was 200)")

    finally:
        try:
            await sb.delete_session(sid)
        except Exception as exc:
            print(f"  [warn] delete_session failed: {type(exc).__name__}: {exc}")
        try:
            await sb.close()
        except Exception:
            pass

    print(f"\n[*] {passed} passed, {failed} failed")
    if failed == 0:
        print("[OK] sandbox is working.")
        return 0
    print("[X] sandbox has problems — see failures above.")
    return 1


if __name__ == "__main__":
    addr = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:6001"
    addr = addr.split("://")[-1]
    h, _, p = addr.partition(":")
    raise SystemExit(asyncio.run(main(h, p or "6001")))
