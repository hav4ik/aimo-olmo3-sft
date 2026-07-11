#!/usr/bin/env python3
"""Convert a HuggingFace messages dataset -> OLMo-core pre-tokenized format
(token_ids_part_*.npy + labels_mask_part_*.npy + dataset_statistics.json + tokenizer/),
using the FAITHFUL open-instruct `olmo_thinker` converter. Shardable across a cluster.

WHY SHARDING (not just more cores): the tokenize step is MEMORY-BANDWIDTH bound — on one box,
~8 cores already saturate it, so adding cores on the *same* box doesn't speed it up. To go faster
you need more aggregate bandwidth: a higher-bandwidth / multi-socket machine, or multiple NODES,
each processing a slice. This script supports both.

Requires (on each node): the open-instruct converter. Two ways to provide it:
  * Docker (default): the `open-instruct-dataprep` image (has the convert script + Olmo deps).
      Get it onto the cluster via `docker pull <your-registry>/open-instruct-dataprep:0.1.0`
      (or `docker save | ssh node docker load`, or rebuild from open-instruct's Dockerfile.dataprep).
  * Python (`--mode python`): a checkout/install of open-instruct on PYTHONPATH, with the convert at
      $OPEN_INSTRUCT/scripts/data/convert_sft_data_for_olmocore.py.

USAGE
  # Single machine:
  python convert_hf_to_olmocore.py --output /out/olmocore --procs $(nproc)

  # Cluster of K nodes — run on node k = 0..K-1, all writing to a SHARED --output:
  python convert_hf_to_olmocore.py --output /shared/olmocore --procs $(nproc) \
      --num-shards K --shard-index k
  # then once, after all nodes finish, merge the per-shard stats:
  python convert_hf_to_olmocore.py --output /shared/olmocore --num-shards K --combine

Each shard writes token_ids_part_<shard>_<n>.npy + labels_mask_part_<shard>_<n>.npy into --output;
OLMo-core globs token_ids_part_*.npy / labels_mask_part_*.npy, so all shards' parts are read together.
"""
import argparse, glob, json, os, shutil, subprocess, sys

DEFAULT_DATASET = "chankhavu/smolmo-sft-v2-seqlen64k"
DEFAULT_TOKENIZER = "allenai/Olmo-3-7B-Think"
DEFAULT_IMAGE = "open-instruct-dataprep:0.1.0"


def combine_stats(output, num_shards):
    """Merge per-shard stats_*.json -> dataset_statistics.json (sums tokens/sequences)."""
    parts = sorted(glob.glob(os.path.join(output, "stats_shard*.json")))
    if not parts:
        sys.exit(f"[combine] no stats_shard*.json in {output}")
    seqs = toks = train = skip = 0
    cfg = {}
    for p in parts:
        d = json.load(open(p))
        o = d.get("overall_statistics", {})
        cfg = d.get("configuration", cfg)
        seqs += o.get("total_instances", 0)
        toks += o.get("total_tokens", 0)
        train += o.get("trainable_tokens", 0)
        skip += o.get("instances_filtered", 0)
    agg = {"total_sequences": seqs, "total_tokens": toks, "total_trainable_tokens": train,
           "trainable_fraction": (train / toks if toks else 0), "num_samples_skipped": skip,
           "max_seq_length": cfg.get("max_sequence_length"), "tokenizer": cfg.get("tokenizer"),
           "chat_template": cfg.get("chat_template"), "shards": len(parts)}
    json.dump(agg, open(os.path.join(output, "dataset_statistics.json"), "w"), indent=2)
    print(f"[combine] {len(parts)} shards -> {seqs:,} seqs, {toks:,} tokens, trainable {agg['trainable_fraction']:.4f}")


def write_messages_shard(dataset, split, num_shards, shard_index, max_examples, out_parquet):
    """Load HF dataset (optionally shard k of K), write its `messages` column to a parquet with
    SMALL row-groups (critical: avoids the OOM that giant row-groups cause in load_dataset)."""
    from datasets import load_dataset
    import pyarrow as pa, pyarrow.parquet as pq
    # accept an HF dataset id OR a local parquet file / dir / glob (handy for offline `hf download`)
    if dataset.endswith(".parquet") or "*" in dataset or os.path.isdir(dataset) or os.path.isfile(dataset):
        files = (sorted(glob.glob(dataset)) if "*" in dataset
                 else [dataset] if os.path.isfile(dataset)
                 else sorted(glob.glob(os.path.join(dataset, "**", "*.parquet"), recursive=True)))
        if not files:
            sys.exit(f"[load] no parquet files matched: {dataset}")
        ds = load_dataset("parquet", data_files=files, split=split)
    else:
        ds = load_dataset(dataset, split=split)
    if num_shards > 1:
        ds = ds.shard(num_shards=num_shards, index=shard_index, contiguous=True)
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    ds = ds.select_columns(["messages"])
    print(f"[shard {shard_index}/{num_shards}] {len(ds):,} examples -> {out_parquet}", flush=True)
    w = None
    for batch in ds.iter(batch_size=2500):                 # ~2500 rows/row-group keeps memory bounded
        tbl = pa.table({"messages": batch["messages"]})
        if w is None:
            w = pq.ParquetWriter(out_parquet, tbl.schema, compression="zstd")
        w.write_table(tbl)
    w.close()


def run_convert(out_parquet, tmpout, tokenizer, max_seq, procs, mode, image, cache_dir):
    """Invoke the faithful open-instruct converter on the shard parquet."""
    conv = "/workspace/open-instruct/scripts/data/convert_sft_data_for_olmocore.py"
    args = [conv,
            "--dataset_mixer_list", out_parquet, "1.0",
            "--tokenizer_name_or_path", tokenizer,
            "--chat_template_name", "olmo_thinker",
            "--output_dir", tmpout,
            "--dataset_local_cache_dir", cache_dir,
            "--max_seq_length", str(max_seq), "--num_examples", "0"]
    env = dict(os.environ, BEAKER_ASSIGNED_CPU_COUNT=str(procs), TOKENIZERS_PARALLELISM="false")
    if mode == "docker":
        root = os.path.commonpath([os.path.dirname(os.path.abspath(p)) for p in (out_parquet, tmpout, cache_dir)])
        cmd = ["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
               "-e", f"BEAKER_ASSIGNED_CPU_COUNT={procs}", "-e", "TOKENIZERS_PARALLELISM=false",
               "-e", f"HF_HOME={cache_dir}/.hf", "-e", f"HOME={cache_dir}",
               "-v", f"{root}:{root}", image, "python"] + args
    else:
        oi = os.environ.get("OPEN_INSTRUCT", "/workspace/open-instruct")
        cmd = [sys.executable, os.path.join(oi, "scripts/data/convert_sft_data_for_olmocore.py")] + args[1:]
    print("[convert]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--split", default="train")
    ap.add_argument("--output", required=True, help="final olmocore dir (shared across nodes for a cluster run)")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--max-seq", type=int, default=65536)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--max-examples", type=int, default=0, help="0=all (use a small N to smoke-test)")
    ap.add_argument("--workdir", default=None, help="scratch dir (default: <output>/_work)")
    ap.add_argument("--mode", choices=["docker", "python"], default="docker")
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--combine", action="store_true", help="merge per-shard stats into dataset_statistics.json and exit")
    a = ap.parse_args()
    os.makedirs(a.output, exist_ok=True)

    if a.combine:
        combine_stats(a.output, a.num_shards)
        return

    workdir = a.workdir or os.path.join(a.output, "_work")
    os.makedirs(workdir, exist_ok=True)
    cache_dir = os.path.join(workdir, f"oi_cache_{a.shard_index:04d}")
    out_parquet = os.path.join(workdir, f"messages_shard{a.shard_index:04d}.parquet")
    tmpout = os.path.join(workdir, f"olmocore_shard{a.shard_index:04d}")
    shutil.rmtree(tmpout, ignore_errors=True); shutil.rmtree(cache_dir, ignore_errors=True)

    write_messages_shard(a.dataset, a.split, a.num_shards, a.shard_index, a.max_examples, out_parquet)
    run_convert(out_parquet, tmpout, a.tokenizer, a.max_seq, a.procs, a.mode, a.image, cache_dir)

    # move outputs into the shared --output with shard-indexed, globally-unique part names
    moved = 0
    for f in sorted(glob.glob(os.path.join(tmpout, "token_ids_part_*.npy"))):
        n = os.path.basename(f).replace("token_ids_part_", "").replace(".npy", "")  # local part number
        shutil.move(f, os.path.join(a.output, f"token_ids_part_{a.shard_index:04d}_{n}.npy"))
        lm = f.replace("token_ids_part_", "labels_mask_part_")
        if os.path.exists(lm):
            shutil.move(lm, os.path.join(a.output, f"labels_mask_part_{a.shard_index:04d}_{n}.npy"))
        cg = f.replace(".npy", ".csv.gz")
        if os.path.exists(cg):
            shutil.move(cg, os.path.join(a.output, f"token_ids_part_{a.shard_index:04d}_{n}.csv.gz"))
        moved += 1
    if os.path.exists(os.path.join(tmpout, "dataset_statistics.json")):
        shutil.copy(os.path.join(tmpout, "dataset_statistics.json"),
                    os.path.join(a.output, f"stats_shard{a.shard_index:04d}.json"))
    if a.shard_index == 0 and os.path.isdir(os.path.join(tmpout, "tokenizer")) \
            and not os.path.isdir(os.path.join(a.output, "tokenizer")):
        shutil.copytree(os.path.join(tmpout, "tokenizer"), os.path.join(a.output, "tokenizer"))
    shutil.rmtree(tmpout, ignore_errors=True); shutil.rmtree(cache_dir, ignore_errors=True)
    os.remove(out_parquet)
    print(f"[shard {a.shard_index}] wrote {moved} part file(s) -> {a.output}")
    if a.num_shards == 1:
        combine_stats(a.output, 1)
    else:
        print(f"[shard {a.shard_index}] done. After ALL shards finish, run with --combine to write dataset_statistics.json")


if __name__ == "__main__":
    main()
