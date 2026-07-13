#!/usr/bin/env python3
"""Dump every session of a trace dir for human/agent audit of grading correctness.

For each session emits one JSON line:
  {pid, sidx, gold, pred, tail, mv}   where
    gold = expected answer, pred = extracted \\boxed{}, tail = last 300 chars of generation,
    mv   = math_verify verdict (True/False/"ERR")

Usage: python build_audit_dump.py <trace_dir> <out.jsonl>
"""
import json, glob, os, sys
from math_verify import parse, verify


def _parse(s):
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    for cand in ("$" + s + "$", s):
        try:
            p = parse(cand)
            if p:
                return p
        except Exception:
            continue
    return None


def mv_verdict(pred, gold):
    gp, pp = _parse(gold), _parse(pred)
    if gp is None or pp is None:
        if pred is not None and str(pred).strip() == str(gold).strip():
            return True
        return "ERR" if pp is None and pred not in (None, "") else False
    try:
        return bool(verify(gp, pp))
    except Exception:
        return "ERR"


def main():
    trace_dir, out = sys.argv[1], sys.argv[2]
    rows = []
    for f in sorted(glob.glob(os.path.join(trace_dir, "*.jsonl"))):
        for line in open(f):
            if not line.strip():
                continue
            s = json.loads(line)
            gen = s.get("generation") or ""
            tail = gen[-300:]
            rows.append({
                "pid": s.get("id"),
                "sidx": s.get("session_index"),
                "gold": str(s.get("expected_answer", s.get("answer"))),
                "pred": None if s.get("boxed") is None else str(s.get("boxed")),
                "tail": tail,
                "mv": mv_verdict(s.get("boxed"), s.get("expected_answer", s.get("answer"))),
            })
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote %d rows -> %s" % (len(rows), out))


if __name__ == "__main__":
    main()
