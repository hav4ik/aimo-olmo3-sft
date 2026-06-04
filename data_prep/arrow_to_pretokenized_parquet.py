#!/usr/bin/env python3
"""Compact an Axolotl prepared (tokenized) dataset into a small, pushable, pre-tokenized
parquet that Axolotl can load natively (skipping re-tokenization at train time).

WHY: `axolotl preprocess` writes its prepared cache as uncompressed HF-datasets arrow with
FOUR per-token int columns (`input_ids` int32, `labels` int64, `attention_mask` int8,
`position_ids` int64) + `length` = ~21 bytes/token. For our 6.08B-token SFT set that's ~120 GB.
We only need to *ship* `input_ids` + `labels` + `attention_mask` (Axolotl re-derives
`position_ids`/`length` cheaply per node). Down-casting `labels` to int32, dropping
`position_ids`/`length`, and writing zstd parquet takes it to ~15 GB (lossless).

Axolotl 0.16.2 detects a dataset that already has `input_ids`+`labels`+`attention_mask`
(`utils/data/wrappers.py:_is_dataset_already_tokenized`) and SKIPS tokenization. `ds_type:
parquet` reads zstd transparently. So the small parquet is a drop-in pre-tokenized input.

USAGE (run in the dataprep image; reads the arrow, writes the parquet):
  python arrow_to_pretokenized_parquet.py --in <axolotl_prepared_dir>/<hash> --out out.parquet

  docker run --rm --user $(id -u):$(id -g) -e HOME=/data/home -v /mnt/data:/data \
    open-instruct-dataprep:0.1.0 python /prep/arrow_to_pretokenized_parquet.py \
      --in /data/.../ax_prepared/<hash> --out /data/.../sft_tokenized.parquet

NOTE: down-casting `labels` int64->int32 is safe iff max token id < 2^31 (Olmo vocab ~100k).
`attention_mask` MUST be kept (the packed flash-attn path derives doc boundaries from it).
The conversion preserves row order; Axolotl reshuffles at train anyway.
"""
import argparse, glob, time
import pyarrow as pa, pyarrow.parquet as pq
from datasets import load_from_disk

SCHEMA = pa.schema([
    ("input_ids", pa.list_(pa.int32())),
    ("labels", pa.list_(pa.int32())),
    ("attention_mask", pa.list_(pa.int8())),
])

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True,
                    help="Axolotl prepared dataset dir (the <hash> dir under dataset_prepared_path)")
    ap.add_argument("--out", required=True, help="output .parquet path")
    ap.add_argument("--batch-size", type=int, default=2000)
    a = ap.parse_args()
    d = a.inp.rstrip("/")
    if glob.glob(f"{d}/*/dataset_info.json"):           # a parent dir was passed; descend
        d = glob.glob(f"{d}/*/")[0].rstrip("/")
    ds = load_from_disk(d).select_columns(["input_ids", "labels", "attention_mask"])
    w = pq.ParquetWriter(a.out, SCHEMA, compression="zstd")
    n, t0 = 0, time.time()
    for b in ds.with_format("arrow").iter(batch_size=a.batch_size):
        tbl = b if isinstance(b, pa.Table) else pa.Table.from_batches([b])
        w.write_table(tbl.select(["input_ids", "labels", "attention_mask"]).cast(SCHEMA))
        n += tbl.num_rows
        if n % 50000 < a.batch_size:
            print(f"  {n}/{ds.num_rows} {time.time()-t0:.0f}s", flush=True)
    w.close()
    print(f"DONE: {n} rows -> {a.out}")

if __name__ == "__main__":
    main()
