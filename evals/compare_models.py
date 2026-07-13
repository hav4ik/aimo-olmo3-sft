#!/usr/bin/env python3
"""
Compare per-problem accuracy across multiple trace directories.

Only shows problems that appear in ALL given directories (overlap set).
Displays per-session accuracy, majority-vote correctness, and head-to-head.

Usage:
  python scripts/compare_models.py traces/gptoss120b-aimofull-v5/ traces/nemotron-aimofull-v1/
  python scripts/compare_models.py --names 120b,Nemotron traces/gptoss120b-aimofull-v5/ traces/nemotron-aimofull-v1/
  python scripts/compare_models.py --all traces/dir1/ traces/dir2/ traces/dir3/
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def _get_prediction(session: dict):
    """Return the parsed answer for a session, tolerating trace-format differences.

    Older collectors stored it under ``predicted_answer``; the current run_eval.py boxed tasks
    store it under ``boxed``. Returns ``None`` when no answer was produced.
    """
    for key in ("predicted_answer", "boxed"):
        val = session.get(key)
        if val is not None:
            return val
    return None


def _get_gen_tokens(session: dict) -> int:
    """Extract generated token count from a session, with fallback estimation."""
    for key in ("num_completion_tokens", "num_generated_tokens"):
        val = session.get(key, 0)
        if val and val > 0:
            return val
    # Fallback: estimate from conversation text (~4 chars per token)
    conv = session.get("conversation", [])
    if not conv:
        trace = session.get("trace", "")
        return len(trace) // 4 if trace else 0
    chars = 0
    for msg in conv:
        if msg.get("role") == "assistant":
            chars += len(msg.get("content", "") or "")
            for tc in msg.get("tool_calls", []):
                chars += len(tc.get("function", {}).get("arguments", ""))
    return chars // 4


def load_traces(trace_dir: Path) -> tuple[dict[str, dict], bool]:
    results = {}
    estimated_tokens = False
    for f in sorted(trace_dir.glob("*.jsonl")):
        pid = f.stem
        sessions = [json.loads(line) for line in open(f) if line.strip()]
        if not sessions:
            continue
        expected = str(sessions[0].get("expected_answer", sessions[0].get("answer", "")))
        n = len(sessions)
        correct = sum(
            1 for s in sessions
            if _get_prediction(s) is not None and str(_get_prediction(s)) == expected
        )
        no_answer = sum(1 for s in sessions if _get_prediction(s) is None)
        truncated = sum(1 for s in sessions if s.get("finish_reason") in ("length", "max_turns"))
        preds = [_get_prediction(s) for s in sessions if _get_prediction(s) is not None]
        wrong = n - correct - no_answer
        maj = Counter(preds).most_common(1)[0][0] if preds else None
        maj_ok = maj is not None and str(maj) == expected
        avg_time = sum(
            s.get("generation_time_seconds", s.get("generation_time", 0)) for s in sessions
        ) / n
        gen_tokens = sum(_get_gen_tokens(s) for s in sessions)
        has_explicit = any(
            (s.get("num_completion_tokens") or 0) > 0 or (s.get("num_generated_tokens") or 0) > 0
            for s in sessions
        )
        if not has_explicit and gen_tokens > 0:
            estimated_tokens = True
        avg_gen_tokens = gen_tokens // n
        tool_calls = sum(s.get("num_tool_calls", 0) for s in sessions)
        tool_errors = sum(s.get("num_tool_errors", 0) for s in sessions)
        avg_tools = tool_calls / n
        results[pid] = {
            "expected": expected,
            "n": n,
            "correct": correct,
            "wrong": wrong,
            "no_answer": no_answer,
            "truncated": truncated,
            "pct": correct / n * 100,
            "maj": maj,
            "maj_ok": maj_ok,
            "avg_time": avg_time,
            "gen_tokens": gen_tokens,
            "avg_gen_tokens": avg_gen_tokens,
            "tool_calls": tool_calls,
            "tool_errors": tool_errors,
            "avg_tools": avg_tools,
        }
    return results, estimated_tokens


def main():
    parser = argparse.ArgumentParser(description="Compare per-problem accuracy across trace dirs")
    parser.add_argument("trace_dirs", nargs="+", help="Trace directories to compare")
    parser.add_argument("--names", help="Comma-separated display names (default: dir basenames)")
    parser.add_argument("--all", action="store_true",
                        help="Show all problems (union), not just overlap")
    args = parser.parse_args()

    dirs = [Path(d) for d in args.trace_dirs]
    if args.names:
        names = args.names.split(",")
        assert len(names) == len(dirs), "Number of names must match number of directories"
    else:
        names = [d.name for d in dirs]

    # Load all traces
    models = {}
    tok_estimated = {}
    for name, d in zip(names, dirs):
        models[name], tok_estimated[name] = load_traces(d)

    # Determine problem set
    all_pid_sets = [set(m.keys()) for m in models.values()]
    if args.all:
        pids = sorted(set().union(*all_pid_sets))
        mode = "union"
    else:
        pids = sorted(set.intersection(*all_pid_sets))
        mode = "overlap"

    if not pids:
        print("No overlapping problems found.")
        return

    # Print legend
    print("=" * 40)
    print("Model Comparison Report")
    print("=" * 40)
    print()
    print("Models:")
    for name, d in zip(names, dirs):
        n_prob = len(models[name])
        n_sess = sum(r["n"] for r in models[name].values())
        print("  %-12s  %s  (%d problems, %d sessions)" % (name, d, n_prob, n_sess))
    print()
    print("Showing: %d problems (%s of all directories)" % (len(pids), mode))
    print()
    print("Column legend:")
    print("  Cor/N  = correct sessions / total sessions")
    print("  Acc    = per-session accuracy")
    print("  F      = no-answer sessions (failed to produce \\boxed{})")
    print("  Truncated = sessions cut at the token/turn cap (finish_reason length/max_turns)")
    print("  Tok    = avg generated tokens per session (~ = estimated from text)")
    print("  Tools  = avg tool calls (Python sandbox) per session")
    print("  MV     = majority vote correct (Y/N)")
    print("  Winner = per-problem winner by majority vote, then per-session accuracy")
    print("           >> = strong win (one correct, other wrong)")
    print()

    # Header
    hdr = "%-8s %6s" % ("PID", "Ans")
    for name in names:
        hdr += " | %-26s %-2s" % (name, "MV")
    hdr += " | Winner"
    print(hdr)
    print("-" * len(hdr))

    # Per-model tallies for summary
    tallies = {n: {"correct": 0, "total": 0, "no_answer": 0, "truncated": 0, "gen_tokens": 0,
                    "tool_calls": 0, "tool_errors": 0,
                    "maj_correct": 0, "problems": 0}
               for n in names}
    wins = {n: 0 for n in names}
    ties = 0

    for pid in pids:
        exp = ""
        for m in models.values():
            if pid in m:
                exp = m[pid]["expected"]
                break

        row = "%-8s %6s" % (pid, exp)
        mv_results = {}
        pct_results = {}

        for name in names:
            m = models[name]
            if pid in m:
                r = m[pid]
                na = r["no_answer"]
                na_str = " %dF" % na if na else "   "
                tok = r["avg_gen_tokens"]
                prefix = "~" if tok_estimated[name] else ""
                tok_str = "%s%dk" % (prefix, tok // 1000) if tok >= 1000 else "%s%d" % (prefix, tok)
                tools_str = "%.0ft" % r["avg_tools"]
                cell = "%d/%d %3.0f%%%s %5s %4s" % (
                    r["correct"], r["n"], r["pct"], na_str, tok_str, tools_str)
                mv = "Y" if r["maj_ok"] else "N"
                row += " | %-26s %-2s" % (cell, mv)
                mv_results[name] = r["maj_ok"]
                pct_results[name] = r["pct"]
                tallies[name]["correct"] += r["correct"]
                tallies[name]["total"] += r["n"]
                tallies[name]["no_answer"] += na
                tallies[name]["truncated"] += r["truncated"]
                tallies[name]["gen_tokens"] += r["gen_tokens"]
                tallies[name]["tool_calls"] += r["tool_calls"]
                tallies[name]["tool_errors"] += r["tool_errors"]
                tallies[name]["maj_correct"] += int(r["maj_ok"])
                tallies[name]["problems"] += 1
            else:
                row += " | %-26s %-2s" % ("-", " ")

        # Winner determination (only if all models have this problem)
        winner = ""
        present = [n for n in names if n in mv_results]
        if len(present) == len(names):
            correct_models = [n for n in names if mv_results[n]]
            wrong_models = [n for n in names if not mv_results[n]]
            if len(correct_models) == len(names):
                best_pct = max(pct_results[n] for n in names)
                worst_pct = min(pct_results[n] for n in names)
                if best_pct - worst_pct < 5:
                    winner = "tie"
                    ties += 1
                else:
                    best = max(names, key=lambda n: pct_results[n])
                    wins[best] += 1
                    winner = best
            elif len(wrong_models) == len(names):
                winner = "ALL WRONG"
                ties += 1
            else:
                if len(correct_models) == 1:
                    w = correct_models[0]
                    wins[w] += 1
                    winner = ">> " + w
                else:
                    winner = "WRONG: " + ",".join(wrong_models)
                    ties += 1

        row += " | %s" % winner
        print(row)

    print("-" * len(hdr))

    # Summary
    print()
    print("Summary (%s, %d problems):" % (mode, len(pids)))
    print()

    # Table-style summary
    fmt_hdr = "  %-12s  %14s  %14s  %18s  %14s  %12s  %18s"
    fmt_row = "  %-12s  %14s  %14s  %18s  %14s  %12s  %18s"
    print(fmt_hdr % ("Model", "Per-session", "Majority", "No-answer", "Truncated", "Avg gen tok", "Avg tools/sess"))
    print("  " + "-" * 112)

    for name in names:
        t = tallies[name]
        if t["total"] == 0:
            continue
        sess_str = "%d/%d %.1f%%" % (t["correct"], t["total"], t["correct"] / t["total"] * 100)
        maj_str = "%d/%d %.1f%%" % (t["maj_correct"], t["problems"], t["maj_correct"] / t["problems"] * 100)
        na_str = "%d/%d (%.1f%%)" % (t["no_answer"], t["total"], t["no_answer"] / t["total"] * 100)
        tr_str = "%d/%d (%.1f%%)" % (t["truncated"], t["total"], t["truncated"] / t["total"] * 100)
        avg_gen = t["gen_tokens"] // t["total"] if t["total"] else 0
        prefix = "~" if tok_estimated[name] else ""
        gen_str = "%s%dk" % (prefix, avg_gen // 1000) if avg_gen >= 1000 else "%s%d" % (prefix, avg_gen)
        avg_tools = t["tool_calls"] / t["total"]
        err_rate = t["tool_errors"] / t["tool_calls"] * 100 if t["tool_calls"] else 0
        tools_str = "%.1f (%.0f%% err)" % (avg_tools, err_rate)
        print(fmt_row % (name, sess_str, maj_str, na_str, tr_str, gen_str, tools_str))

    # Head-to-head
    if len(names) >= 2:
        print()
        parts = []
        for n in names:
            if wins[n]:
                parts.append("%s wins %d" % (n, wins[n]))
        if ties:
            parts.append("ties %d" % ties)
        print("  Head-to-head: %s" % ", ".join(parts))


if __name__ == "__main__":
    main()
