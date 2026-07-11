import os, glob, sys, shutil, collections
sys.path.insert(0,"/data/proof-redesign/scripts")
import build_mix as B
import pyarrow as pa, pyarrow.parquet as pq
DATA, PARTS, OUT = B.DATA, B.PARTS, B.OUT
B.init_worker()
def src0(p):
    return pq.ParquetFile(p).read_row_group(0, columns=["source"]).column("source")[0].as_py()
# 1) drop corrupt fineproofs shards (race-collided) wherever they are
dropped=0
for p in glob.glob(DATA+"/*.parquet")+glob.glob(PARTS+"/fineproofs*.parquet"):
    try:
        if src0(p)=="fineproofs": os.remove(p); dropped+=1
    except Exception: pass
print("dropped corrupt fineproofs shards:", dropped)
# 2) reprocess fineproofs correctly (unique names per file)
fp=0
for fi,f in enumerate(sorted(glob.glob("/data/FineProofs-SFT/data/*.parquet"))):
    pfh=pq.ParquetFile(f)
    for rg in range(pfh.metadata.num_row_groups):
        rows=pfh.read_row_group(rg, columns=B.SRC["fineproofs"]["cols"]).to_pylist()
        recs,_,_,_=B.process_rows("fineproofs", rows)
        if recs:
            outp=os.path.join(PARTS, f"fineproofs_f{fi}_{rg:05d}.parquet")
            pq.write_table(pa.Table.from_pylist(recs, schema=B.SCHEMA), outp, compression="zstd")
            fp+=len(recs)
print("reprocessed fineproofs rows:", fp)
# 3) gather all shards, two-phase rename into DATA
allsh = sorted(glob.glob(DATA+"/*.parquet")) + sorted(glob.glob(PARTS+"/*.parquet"))
N=len(allsh)
tmp=[]
for i,p in enumerate(allsh):
    t=os.path.join(DATA, f"_tmp_{i:06d}.parquet"); shutil.move(p,t); tmp.append(t)
for k,t in enumerate(tmp):
    os.rename(t, os.path.join(DATA, f"train-{k:05d}-of-{N:05d}.parquet"))
shutil.rmtree(PARTS, ignore_errors=True)
print("finalized shards:", N)
# 4) recompute totals from shards + write README
agg=collections.defaultdict(lambda: collections.Counter()); tottok=0; totrows=0
for k in range(N):
    t=pq.read_table(os.path.join(DATA, f"train-{k:05d}-of-{N:05d}.parquet"), columns=["source","num_tokens"])
    for s,n in zip(t.column("source").to_pylist(), t.column("num_tokens").to_pylist()):
        agg[s]["rows"]+=1; agg[s]["tokens"]+=n; 
    tottok+=sum(t.column("num_tokens").to_pylist()); totrows+=t.num_rows
with open(os.path.join(OUT,"README.md"),"w") as f:
    f.write("# smolmo-math-cot-sft\n\nMath chain-of-thought SFT for Olmo-3.1-Think (native ChatML/`<think>` format).\n\n## Mixture\n\n")
    f.write("| source | rows | tokens |\n|---|---|---|\n")
    for s,a in sorted(agg.items()): f.write(f"| {s} | {a['rows']:,} | {a['tokens']:,} |\n")
    f.write(f"| **TOTAL** | **{totrows:,}** | **{tottok:,}** |\n\n")
    f.write(f"{N} shards `data/train-XXXXX-of-{N:05d}.parquet` (zstd). Schema: {', '.join(B.SCHEMA.names)}.\n")
    f.write("Filters: every assistant turn has `<think>`; every sample <65536 tokens (full Olmo render). ")
    f.write("Format: `<think> ` open / `\\n</think>\\n\\n` close; `<function_calls>` pythonic tools. See INFERENCE_NOTES.md.\n")
    f.write("NOTE: shards are source-ordered (NOT yet shuffled) — full reshuffle pending.\n")
print(f"TOTAL rows={totrows:,} tokens={tottok:,} shards={N}")
print("per-source:", {s:dict(a) for s,a in agg.items()})
