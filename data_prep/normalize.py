#!/usr/bin/env python3
"""Resolve a training-data input spec into ONE normalized `messages` parquet that BOTH
frameworks consume offline — OLMo-core tokenizes it via open-instruct, Axolotl reads it as a
local parquet — so a single input produces byte-identical examples on both sides.

Input forms (any of):
  - HF dataset id:                    --input allenai/tulu-3-sft-personas-math
  - parquet files inside an HF repo:  --input org/name --data-files "sft/*.parquet"
  - local parquet file(s) / glob:     --input "/data/in/*.parquet"   (or comma-list, or repeat --input)
  - hf:// parquet glob:               --input "hf://datasets/org/name/sft/*.parquet"

Output: a parquet with a single `messages` column = list[{role, content}]. If the source
already has it, pass through (rename via --messages-field). Otherwise build turns from
--prompt-field/--response-field (+ optional --system-field).
"""
import argparse, glob, os, sys
from datasets import load_dataset, concatenate_datasets


def _looks_like_parquet(spec: str) -> bool:
    return spec.startswith("hf://") or spec.endswith(".parquet") or "*" in spec or os.path.exists(spec)


def load_one(spec: str, split: str, data_files):
    # parquets inside a named HF repo, e.g. --input org/name --data-files "sft/*.parquet"
    if data_files and not _looks_like_parquet(spec):
        return load_dataset(spec, data_files=data_files, split=split)
    if _looks_like_parquet(spec):
        files = []
        for part in spec.split(","):
            part = part.strip()
            files += sorted(glob.glob(part)) if ("*" in part and not part.startswith("hf://")) else [part]
        if not files:
            sys.exit(f"[normalize] no files matched: {spec}")
        return load_dataset("parquet", data_files=files, split=split)
    return load_dataset(spec, split=split)  # plain HF dataset id


def to_messages(ex, a):
    if a.messages_field in ex and ex[a.messages_field]:
        return {"messages": ex[a.messages_field]}
    msgs = []
    if a.system_field and ex.get(a.system_field):
        msgs.append({"role": "system", "content": ex[a.system_field]})
    msgs.append({"role": "user", "content": ex[a.prompt_field]})
    msgs.append({"role": "assistant", "content": ex[a.response_field]})
    return {"messages": msgs}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, action="append",
                    help="HF id, parquet glob/list, or hf:// glob. Repeat for multiple sources.")
    ap.add_argument("--data-files", default=None, help="parquet glob INSIDE the HF repo named by --input")
    ap.add_argument("--split", default="train")
    ap.add_argument("--output", required=True, help="output .parquet path")
    ap.add_argument("--messages-field", default="messages", help="source column already in [{role,content}] form")
    ap.add_argument("--prompt-field", default=None, help="build messages from this (user) + --response-field")
    ap.add_argument("--response-field", default=None)
    ap.add_argument("--system-field", default=None)
    ap.add_argument("--max-examples", type=int, default=0, help="0 = all")
    a = ap.parse_args()

    parts = [load_one(s, a.split, a.data_files) for s in a.input]
    ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    if a.max_examples:
        ds = ds.select(range(min(a.max_examples, len(ds))))

    cols = ds.column_names
    if a.messages_field not in cols and not (a.prompt_field and a.response_field):
        sys.exit(f"[normalize] no '{a.messages_field}' column (have {cols}); "
                 f"pass --prompt-field/--response-field (and optionally --system-field) to map.")
    ds = ds.map(lambda ex: to_messages(ex, a), remove_columns=[c for c in cols if c != "messages"],
                desc="normalize->messages")

    sample = ds[0]["messages"]
    assert isinstance(sample, list) and sample and "role" in sample[0] and "content" in sample[0], \
        f"[normalize] bad messages schema after mapping: {sample[:1]}"
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    ds.to_parquet(a.output)
    print(f"[normalize] {len(ds)} examples -> {a.output} | first-row roles: {[m['role'] for m in sample]}")


if __name__ == "__main__":
    main()
