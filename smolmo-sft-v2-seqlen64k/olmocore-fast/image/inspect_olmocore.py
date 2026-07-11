#!/usr/bin/env python3
"""Inspect an OLMo-core pretokenized output dir WITHOUT uploading.

Prints dataset_statistics.json summary, file inventory, sanity checks, and (optionally) decodes
the first N sequences showing the masked (prompt/tool) vs trained (assistant) spans so you can
eyeball that masking landed correctly before you upload.

Usage: inspect_olmocore.py <output_dir> [--show N]
Reads token_ids_part_*.npy (uint32, raw memmap via np.fromfile) + labels_mask_part_*.npy (bool)
+ token_ids_part_*.csv.gz (document boundaries) + tokenizer/.
"""
import argparse, glob, gzip, json, os, sys
import numpy as np


def _concat(d, name, dt):
    fs = sorted(glob.glob(os.path.join(d, f"{name}_part_*.npy")))
    if not fs:
        return np.array([], dtype=dt), fs
    return np.concatenate([np.fromfile(f, dtype=dt) for f in fs]), fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--show", type=int, default=0, help="decode+role-align the first N sequences")
    a = ap.parse_args()
    d = a.output_dir

    sj = os.path.join(d, "dataset_statistics.json")
    if os.path.exists(sj):
        s = json.load(open(sj))
        o = s.get("overall_statistics", {}); c = s.get("configuration", {})
        print("=== dataset_statistics.json ===")
        print(f"  sequences      : {o.get('total_instances'):,}")
        print(f"  tokens         : {o.get('total_tokens'):,}")
        print(f"  trainable      : {o.get('trainable_tokens'):,} ({o.get('trainable_percentage',0):.2f}%)")
        print(f"  filtered/skipped: {o.get('instances_filtered')}")
        print(f"  max_seq_length : {c.get('max_sequence_length')}   chat_template: {c.get('chat_template')}")
        print(f"  tokenizer      : {c.get('tokenizer')}")
    else:
        print("WARNING: no dataset_statistics.json found", file=sys.stderr)

    ids, idf = _concat(d, "token_ids", np.uint32)
    msk, mkf = _concat(d, "labels_mask", np.bool_)
    print(f"\n=== files ===\n  token_ids parts : {len(idf)}  total tokens={len(ids):,}")
    print(f"  labels_mask parts: {len(mkf)}  total={len(msk):,}")
    # sanity checks
    ok = True
    if len(ids) != len(msk):
        print("  !! token_ids and labels_mask length MISMATCH", file=sys.stderr); ok = False
    if len(ids):
        mx = int(ids.max())
        print(f"  max token id    : {mx}  (vocab ~100278; {'OK' if mx < 100278 else 'OUT OF RANGE!'})")
        frac = msk.mean() if len(msk) else 0
        print(f"  trainable frac  : {frac:.4f}  ({'OK' if 0.7 < frac < 0.99 else 'CHECK'})")
    print(f"  sanity          : {'OK' if ok else 'PROBLEM — see above'}")

    if a.show and len(ids):
        bounds = []
        for f in sorted(glob.glob(os.path.join(d, "token_ids_part_*.csv.gz"))):
            with gzip.open(f, "rt") as fh:
                for ln in fh:
                    x = ln.strip().split(",")
                    if len(x) == 2:
                        bounds.append((int(x[0]), int(x[1])))
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(os.path.join(d, "tokenizer"))
        except Exception as e:
            print(f"\n(could not load tokenizer to decode: {e})"); return
        print(f"\n=== first {a.show} sequence(s): per-turn masked/trained ===")
        for di, (s, e) in enumerate(bounds[: a.show]):
            seq, m = ids[s:e], msk[s:e]
            print(f"\n-- SEQ {di}: {e-s} tokens, trained={int(m.sum())} masked={int((~m).sum())} --")
            i = 0
            while i < len(seq):
                if seq[i] == 100264:  # <|im_start|>
                    j = i + 1
                    while j < len(seq) and seq[j] not in (100265, 100257):
                        j += 1
                    role = tok.decode(seq[i+1:min(i+6, len(seq))]).split("\n")[0][:14]
                    frac = m[i:j+1].mean()
                    print(f"   <|im_start|>{role:<14} [{i}:{j+1}] trained_frac={frac:.2f} -> "
                          f"{'TRAINED' if frac > 0.5 else 'masked'}")
                    i = j + 1
                else:
                    i += 1


if __name__ == "__main__":
    main()
