#!/usr/bin/env python3
"""Multi-node OLMo-core conversion — split by parquet shard ids.

Node PART_NUM of NUM_PARTS converts the contiguous shard range [k*N//K, (k+1)*N//K) of the dataset's
N parquet shards (N auto-detected from the repo), DIRECTLY (no re-write), deterministically (no
internal shuffle), and writes node-prefixed part files so OLMo-core globs them together:
    token_ids_part_{part:03d}_{local}.npy   labels_mask_part_{part:03d}_{local}.npy
    token_ids_part_{part:03d}_{local}.csv.gz   stats_part_{part:03d}.json   (tokenizer/ from part 0)

Per-sequence tokens/masks are IDENTICAL to the stock olmo_thinker converter; only sequence ORDER is
deterministic (shard order) instead of seed-shuffled. After all nodes finish + upload, run
--merge-stats to write the combined dataset_statistics.json (fails if any node's part is missing).

Usage:
  convert_node.py --part-num k --num-parts K --dataset <id> --work /out --procs 48
  convert_node.py --merge-stats --num-parts K --work /out
"""
import argparse, glob, json, os, re, shutil, subprocess, sys

OPEN_INSTRUCT = os.environ.get("OPEN_INSTRUCT", "/workspace/open-instruct")
_SHARD_RE = re.compile(r"data/train-\d+-of-\d+\.parquet$")


def shard_range(part, nparts, total):
    return part * total // nparts, (part + 1) * total // nparts


def list_repo_shards(dataset):
    """Return (canonical_complete_shard_list, total). The total is taken from the AUTHORITATIVE
    '-of-NNNNN' encoded in the filenames, NOT from len(listing) — so a partial/paginated repo
    listing (which made different nodes disagree, e.g. 1024 vs 768) cannot drop or misalign shards.
    We then build the COMPLETE canonical file list from that total and download by name."""
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(dataset, repo_type="dataset")
    shards = sorted(f for f in files if _SHARD_RE.match(f))
    if not shards:
        sys.exit(f"[fatal] no data/train-*-of-*.parquet shards found in {dataset}")
    m = re.search(r"train-(?P<idx>\d+)-of-(?P<total>\d+)\.parquet$", shards[0])
    if not m:
        sys.exit(f"[fatal] unexpected shard name: {shards[0]}")
    total = int(m.group("total")); iw = len(m.group("idx")); tw = len(m.group("total"))
    base = shards[0][:shards[0].rindex("train-")]                      # e.g. "data/"
    canon = [f"{base}train-{i:0{iw}d}-of-{total:0{tw}d}.parquet" for i in range(total)]
    if len(shards) != total:                                          # listing was incomplete
        print(f"[warn] repo listing returned {len(shards)} of {total} shards (paginated/partial); "
              f"using authoritative total={total} from filenames and downloading by name.", file=sys.stderr)
    return canon, total


def download_subset(dataset, rel_files, local_dir, workers):
    from huggingface_hub import snapshot_download
    snapshot_download(dataset, repo_type="dataset", allow_patterns=rel_files,
                      local_dir=local_dir, max_workers=workers)


def run_convert(comma_list, out_dir, cache_dir, procs):
    env = dict(os.environ, BEAKER_ASSIGNED_CPU_COUNT=str(procs),
               TOKENIZERS_PARALLELISM="false", SMOLMO_NO_SHUFFLE="1")  # deterministic order
    cmd = [sys.executable, f"{OPEN_INSTRUCT}/scripts/data/convert_sft_data_for_olmocore.py",
           "--dataset_mixer_list", comma_list, "1.0",
           "--tokenizer_name_or_path", "allenai/Olmo-3-7B-Think",
           "--chat_template_name", "olmo_thinker",
           "--output_dir", out_dir, "--dataset_local_cache_dir", cache_dir,
           "--max_seq_length", "65536", "--num_examples", "0"]
    print("[convert]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def rename_parts(tmpout, final_dir, part):
    """Move token_ids_part_NNNN.* -> token_ids_part_{part:03d}_NNNN.* (+ labels_mask, csv.gz).
    Purges any pre-existing parts for THIS node first (kills stale-part mixing on re-runs)."""
    os.makedirs(final_dir, exist_ok=True)
    pre = f"{part:03d}_"
    for stale in glob.glob(os.path.join(final_dir, f"token_ids_part_{pre}*")) \
            + glob.glob(os.path.join(final_dir, f"labels_mask_part_{pre}*")) \
            + glob.glob(os.path.join(final_dir, f"stats_part_{part:03d}.json")):
        os.remove(stale)
    moved = 0
    for f in sorted(glob.glob(os.path.join(tmpout, "token_ids_part_*.npy"))):
        local = os.path.basename(f)[len("token_ids_part_"):-len(".npy")]
        new = f"token_ids_part_{part:03d}_{local}"
        shutil.move(f, os.path.join(final_dir, new + ".npy"))
        lm = f.replace("token_ids_part_", "labels_mask_part_")
        if os.path.exists(lm):
            shutil.move(lm, os.path.join(final_dir, f"labels_mask_part_{part:03d}_{local}.npy"))
        cg = f[:-len(".npy")] + ".csv.gz"
        if os.path.exists(cg):
            shutil.move(cg, os.path.join(final_dir, new + ".csv.gz"))
        moved += 1
    st = os.path.join(tmpout, "dataset_statistics.json")
    if os.path.exists(st):
        shutil.copy(st, os.path.join(final_dir, f"stats_part_{part:03d}.json"))
    tdir = os.path.join(tmpout, "tokenizer")
    if part == 0 and os.path.isdir(tdir) and not os.path.isdir(os.path.join(final_dir, "tokenizer")):
        shutil.copytree(tdir, os.path.join(final_dir, "tokenizer"))
    return moved


def merge_stats(final_dir, nparts):
    parts = sorted(glob.glob(os.path.join(final_dir, "stats_part_*.json")))
    if not parts:
        sys.exit(f"[merge] no stats_part_*.json in {final_dir}")
    seqs = toks = train = skip = 0
    cfg = {}
    for p in parts:
        d = json.load(open(p)); o = d.get("overall_statistics", {}); cfg = d.get("configuration", cfg)
        seqs += o.get("total_instances", 0); toks += o.get("total_tokens", 0)
        train += o.get("trainable_tokens", 0); skip += o.get("instances_filtered", 0)
    agg = {"configuration": cfg, "overall_statistics": {
        "total_instances": seqs, "total_tokens": toks, "trainable_tokens": train,
        "trainable_percentage": (train / toks * 100) if toks else 0,
        "instances_filtered": skip, "average_sequence_length": (toks / seqs) if seqs else 0,
        "num_parts_merged": len(parts)}}
    json.dump(agg, open(os.path.join(final_dir, "dataset_statistics.json"), "w"), indent=2)
    print(f"[merge] {len(parts)}/{nparts} parts -> {seqs:,} seq, {toks:,} tok, "
          f"trainable {agg['overall_statistics']['trainable_percentage']:.2f}%")
    if len(parts) != nparts:                         # completeness gate: FAIL, don't silently undercount
        sys.exit(f"[merge] FATAL: found {len(parts)} parts but expected {nparts} — a node did not "
                 f"finish/upload. dataset_statistics.json written but is INCOMPLETE; do not train yet.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part-num", type=int)
    ap.add_argument("--num-parts", type=int, required=True)
    ap.add_argument("--shard-start", type=int, default=None, help="explicit hard-coded start shard index")
    ap.add_argument("--shard-end", type=int, default=None, help="explicit hard-coded end (exclusive)")
    ap.add_argument("--dataset", default=os.environ.get("DATASET", "chankhavu/smolmo-sft-v2-seqlen64k"))
    ap.add_argument("--work", default=os.environ.get("WORK", "/out"))
    ap.add_argument("--output", default=None)
    ap.add_argument("--procs", type=int, default=int(os.environ.get("PROCS", os.cpu_count())))
    ap.add_argument("--dl-workers", type=int, default=int(os.environ.get("DL_WORKERS", 16)))
    ap.add_argument("--merge-stats", action="store_true")
    a = ap.parse_args()
    out = a.output or os.path.join(a.work, "olmocore")
    os.makedirs(out, exist_ok=True)

    if a.merge_stats:
        merge_stats(out, a.num_parts); return

    assert a.part_num is not None and 0 <= a.part_num < a.num_parts, "--part-num must be in [0, num-parts)"
    shards, total = list_repo_shards(a.dataset)      # authoritative total from filenames, complete list
    if a.shard_start is not None or a.shard_end is not None:
        start = a.shard_start if a.shard_start is not None else 0
        end = a.shard_end if a.shard_end is not None else total
        mode = "HARD-CODED range"
    else:
        if a.num_parts > total:
            sys.exit(f"[fatal] num-parts ({a.num_parts}) > shard count ({total}); use fewer nodes")
        start, end = shard_range(a.part_num, a.num_parts, total)
        mode = "auto-split"
    assert 0 <= start < end <= total, f"bad shard range [{start}:{end}) for total {total}"
    rel = shards[start:end]
    print(f"[node {a.part_num}/{a.num_parts}] repo has {total} shards; this node = [{start}:{end}) "
          f"= {len(rel)} files ({mode})", flush=True)

    ds_dir = os.path.join(a.work, "ds")
    download_subset(a.dataset, rel, ds_dir, a.dl_workers)
    local = sorted(os.path.join(ds_dir, r) for r in rel)
    missing = [p for p in local if not os.path.exists(p)]
    assert not missing, f"missing after download: {missing[:3]}"
    comma_list = ",".join(local)

    tmpout = os.path.join(a.work, f"_tmp_part_{a.part_num:03d}")
    cache = os.path.join(a.work, f"_cache_part_{a.part_num:03d}")
    shutil.rmtree(tmpout, ignore_errors=True); shutil.rmtree(cache, ignore_errors=True)
    run_convert(comma_list, tmpout, cache, a.procs)
    n = rename_parts(tmpout, out, a.part_num)
    shutil.rmtree(tmpout, ignore_errors=True); shutil.rmtree(cache, ignore_errors=True)
    print(f"[node {a.part_num}] wrote {n} part file(s) -> {out}")


if __name__ == "__main__":
    main()
