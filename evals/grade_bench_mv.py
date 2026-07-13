#!/usr/bin/env python3
"""Robust re-grader using HuggingFace math_verify (the MathArena-standard verifier).

Unlike grade_bench.py (near-exact string match, which UNDER-COUNTS symbolic HMMT
answers), this uses math_verify.verify() so \\frac{9}{\\sqrt{23}} == \\frac{9\\sqrt{23}}{23},
8\\sqrt{10} == \\sqrt{640}, reordered roots, latex-vs-plain, etc. all grade correctly.

Reads run_eval boxed traces: each jsonl line has {"boxed": <pred>, "expected_answer": <gold>,
"finish_reason": ...}. Reports Avg@k (per-session), maj@k (equivalence-clustered majority
vote), and truncation. Use for HMMT (mixed integer/latex answers); AIME (pure integers) is
unaffected but grades identically here.

Usage:
  python grade_bench_mv.py traces/bench_0207full_proofpilot_hmmt_feb2025_tool [--label NAME]
"""
import argparse, glob, json, os, sys
from collections import Counter

from math_verify import parse, verify


def _parse(s):
    """Parse a raw answer/prediction string into math_verify's comparable form."""
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    # Wrap in $...$ so math_verify treats it as a latex/math expression; fall back to raw.
    for cand in ("$" + s + "$", s):
        try:
            p = parse(cand)
            if p:
                return p
        except Exception:
            continue
    return None


def _eq(pred, gold):
    gp, pp = _parse(gold), _parse(pred)
    if gp is None or pp is None:
        # last resort: exact string
        return pred is not None and str(pred).strip() == str(gold).strip()
    try:
        return bool(verify(gp, pp))
    except Exception:
        return False


def _maj_vote(preds, gold):
    """Cluster predictions by math-equivalence, return whether the largest cluster is correct."""
    preds = [p for p in preds if p is not None]
    if not preds:
        return False, None
    clusters = []  # list of [representative, count, is_correct]
    for p in preds:
        placed = False
        for c in clusters:
            if _eq(p, c[0]):
                c[1] += 1
                placed = True
                break
        if not placed:
            clusters.append([p, 1, _eq(p, gold)])
    clusters.sort(key=lambda c: -c[1])
    top = clusters[0]
    return top[2], top[0]


def grade_dir(trace_dir):
    files = sorted(glob.glob(os.path.join(trace_dir, "*.jsonl")))
    n_sess = n_correct = n_trunc = 0
    prob_maj = prob_maj_ok = 0
    for f in files:
        sessions = [json.loads(l) for l in open(f) if l.strip()]
        if not sessions:
            continue
        gold = sessions[0].get("expected_answer", sessions[0].get("answer"))
        preds = []
        for s in sessions:
            pred = s.get("boxed", s.get("predicted_answer"))
            preds.append(pred)
            n_sess += 1
            if _eq(pred, gold):
                n_correct += 1
            if s.get("finish_reason") in ("length", "max_turns"):
                n_trunc += 1
        ok, _ = _maj_vote(preds, gold)
        prob_maj += 1
        prob_maj_ok += int(ok)
    return n_correct, n_sess, prob_maj_ok, prob_maj, n_trunc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("--label")
    args = ap.parse_args()
    c, n, mo, mp, tr = grade_dir(args.trace_dir)
    print("=== %s ===" % (args.label or args.trace_dir))
    if n == 0:
        print("(no sessions)")
        return
    print("Avg@k (per-session): %d/%d = %.1f%%" % (c, n, c / n * 100))
    print("maj@k              : %d/%d = %.1f%%" % (mo, mp, mo / mp * 100))
    print("truncated sessions : %d/%d = %.1f%%" % (tr, n, tr / n * 100))


if __name__ == "__main__":
    main()
