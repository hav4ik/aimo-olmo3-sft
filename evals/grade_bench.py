#!/usr/bin/env python3
"""Grade run_eval.py boxed traces: Avg@k (per-session) + maj@k, per-benchmark.

run_eval writes {boxed, expected_answer, ...} per session; this compares them with robust
math equivalence. Primary check is HuggingFace **math_verify** (the MathArena-standard verifier),
which correctly handles AIME integers AND HMMT symbolic answers (fractions, radicals, pi,
factorials, reordered root-sets, latex-vs-plain, rationalized-vs-not). Audited 2026-07-03 against
720 HMMT sessions via 9 independent sub-agents: 0 grading errors. If math_verify is unavailable,
falls back to the older string/sympy normalization (which UNDER-COUNTS symbolic answers — install
math_verify: pip install math-verify)."""
import argparse, glob, json, sys
from collections import Counter, defaultdict

# --- Primary equivalence: HuggingFace math_verify (MathArena-standard) ---
try:
    from math_verify import parse as _mv_parse, verify as _mv_verify
    _HAVE_MV = True
except Exception:
    _HAVE_MV = False

def _mv_to_expr(s):
    if s is None: return None
    s = str(s).strip()
    if not s: return None
    for cand in ("$" + s + "$", s):
        try:
            p = _mv_parse(cand)
            if p: return p
        except Exception:
            continue
    return None

def _mv_equal(pred, exp):
    """math_verify equivalence; returns None if it cannot parse either side (caller falls back)."""
    gp, pp = _mv_to_expr(exp), _mv_to_expr(pred)
    if gp is None or pp is None:
        return None
    try:
        return bool(_mv_verify(gp, pp))
    except Exception:
        return None

def _norm(x):
    if x is None: return None
    s = str(x).strip()
    for a, b in [("\\left",""),("\\right",""),("\\!",""),("\\,",""),("\\ ",""),("$",""),
                 ("\\text{","{"),(" ",""),(",","")]:
        s = s.replace(a, b)
    s = s.strip().strip("{}").strip()
    return s or None

def _equal(pred, exp):
    # Primary: math_verify (validated); only fall through when it can't parse a side.
    if _HAVE_MV:
        mv = _mv_equal(pred, exp)
        if mv is not None:
            return mv
    p, e = _norm(pred), _norm(exp)
    if p is None or e is None: return False
    if p == e: return True
    # numeric / symbolic equivalence (fractions, negatives, simple expressions)
    try:
        from sympy import simplify, Rational, sympify
        from sympy.parsing.latex import parse_latex  # optional
    except Exception:
        try:
            return abs(float(p) - float(e)) < 1e-9
        except Exception:
            return False
    def _to_expr(z):
        for fn in (lambda t: sympify(t.replace("\\frac","").replace("\\dfrac","")), ):
            try: return fn(z)
            except Exception: pass
        try: return sympify(z)
        except Exception: return None
    ep, ee = _to_expr(p), _to_expr(e)
    if ep is not None and ee is not None:
        try:
            return simplify(ep - ee) == 0
        except Exception:
            return False
    try:
        return abs(float(p) - float(e)) < 1e-9
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    by_prob = defaultdict(list)
    for f in glob.glob(f"{args.trace_dir}/*.jsonl") + glob.glob(f"{args.trace_dir}/*.json"):
        for line in open(f):
            line = line.strip()
            if not line: continue
            try: d = json.loads(line)
            except Exception: continue
            if "id" not in d: continue
            pred = d.get("boxed", d.get("predicted_answer"))
            by_prob[d["id"]].append((pred, d.get("expected_answer"), d.get("finish_reason")))
    if not by_prob:
        print(f"no traces in {args.trace_dir}"); return 1
    tot_sess = tot_correct = maj_correct = n = trunc = 0
    rows = []
    for pid, sess in sorted(by_prob.items()):
        exp = sess[0][1]
        c = sum(1 for p, _, _ in sess if _equal(p, exp))
        # Majority vote: cluster raw preds by math-equivalence (not string), grade the top cluster.
        raw_preds = [p for p, _, _ in sess if p is not None]
        clusters = []  # [representative, count, is_correct]
        for p in raw_preds:
            for cl in clusters:
                if _equal(p, cl[0]):
                    cl[1] += 1
                    break
            else:
                clusters.append([p, 1, _equal(p, exp)])
        clusters.sort(key=lambda cl: -cl[1])
        mc = int(bool(clusters) and clusters[0][2])
        tr = sum(1 for _, _, fr in sess if fr in ("length", "max_turns"))
        n += 1; tot_sess += len(sess); tot_correct += c; maj_correct += mc; trunc += tr
        rows.append((pid, exp, len(sess), c, mc, tr))
    print(f"\n=== {args.label or args.trace_dir} ===")
    print(f"{'id':<14}{'exp':>8}{'sess':>5}{'ok':>4}{'maj':>4}{'trunc':>6}")
    for pid, exp, s, c, mc, tr in rows:
        print(f"{pid:<14}{str(exp):>8}{s:>5}{c:>4}{'Y' if mc else 'N':>4}{tr:>6}")
    avg = tot_correct / tot_sess if tot_sess else 0
    maj = maj_correct / n if n else 0
    print(f"\nAvg@k (per-session): {tot_correct}/{tot_sess} = {avg:.1%}")
    print(f"maj@k              : {maj_correct}/{n} = {maj:.1%}")
    print(f"truncated sessions : {trunc}/{tot_sess} = {trunc/tot_sess:.1%}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
