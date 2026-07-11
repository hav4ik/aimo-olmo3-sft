#!/usr/bin/env python3
"""Full row-shuffle of all shards (existing data/ + new _parts/) with per-source token budgets.
Downsample math_notool->6B, math_withtool->10B via random keep-prob (keeps full shuffle); take-all else.
Scatter rows to random buckets, then per-bucket shuffle -> uniform final shards in data/."""
import os, glob, sys, random, shutil, collections
sys.path.insert(0,"/data/proof-redesign/scripts")
import build_mix as B
import pyarrow as pa, pyarrow.parquet as pq
DATA, PARTS, OUT, SCH = B.DATA, B.PARTS, B.OUT, B.SCHEMA
SEED=20260609; NB=128; TARGET=3000
# keep-probability for budgeted sources (budget / built-total); others keep all
KEEP={"math_notool": 6.0/12.120157268, "math_withtool": 10.0/12.012785837}
rng=random.Random(SEED)
inputs=sorted(glob.glob(DATA+"/*.parquet"))+sorted(glob.glob(PARTS+"/*.parquet"))
print(f"input shards: {len(inputs)} (data/={len(glob.glob(DATA+'/*.parquet'))} parts/={len(glob.glob(PARTS+'/*.parquet'))})",flush=True)
SC=OUT+"/_scatter"; os.makedirs(SC,exist_ok=True)
W=[pq.ParquetWriter(f"{SC}/b{i:03d}.parquet",SCH,compression="zstd") for i in range(NB)]
buf=[[] for _ in range(NB)]
def fl(i):
    if buf[i]: W[i].write_table(pa.Table.from_pylist(buf[i],schema=SCH)); buf[i]=[]
seen=collections.Counter(); kept=collections.Counter()
for n,sh in enumerate(inputs):
    for row in pq.read_table(sh).to_pylist():
        s=row["source"]; seen[s]+=1
        p=KEEP.get(s)
        if p is not None and rng.random()>=p: continue
        kept[s]+=1; b=rng.randrange(NB); buf[b].append(row)
        if len(buf[b])>=400: fl(b)
    if (n+1)%100==0: print(f"  scattered {n+1}/{len(inputs)} shards",flush=True)
for i in range(NB): fl(i); W[i].close()
print("scatter done. kept/seen per source:",{s:(kept[s],seen[s]) for s in seen},flush=True)
# gather: per bucket shuffle, write final shards
ND=OUT+"/_data_new"; os.makedirs(ND,exist_ok=True)
k=0; tottok=collections.Counter(); totrows=collections.Counter()
parts=[]
for i in range(NB):
    rows=pq.read_table(f"{SC}/b{i:03d}.parquet").to_pylist()
    rng.shuffle(rows)
    for j in range(0,len(rows),TARGET):
        chunk=rows[j:j+TARGET]
        pq.write_table(pa.Table.from_pylist(chunk,schema=SCH),f"{ND}/p{k:05d}.parquet",compression="zstd"); k+=1
        for r in chunk: tottok[r["source"]]+=r["num_tokens"]; totrows[r["source"]]+=1
N=k
# rename to final, replace data/
shutil.rmtree(DATA); os.makedirs(DATA)
for idx,f in enumerate(sorted(glob.glob(ND+"/*.parquet"))):
    os.rename(f, f"{DATA}/train-{idx:05d}-of-{N:05d}.parquet")
shutil.rmtree(SC,ignore_errors=True); shutil.rmtree(ND,ignore_errors=True); shutil.rmtree(PARTS,ignore_errors=True)
GT=sum(tottok.values()); GR=sum(totrows.values())
print(f"\nFINAL shuffled mix: {GR:,} rows, {GT:,} tokens, {N} shards")
for s in sorted(tottok): print(f"  {s:14s} rows={totrows[s]:>9,} tokens={tottok[s]/1e9:6.2f}B")
# README
with open(OUT+"/README.md","w") as f:
    f.write("# smolmo-math-cot-sft\n\nMath CoT SFT for Olmo-3.1-Think (native `<think>`/ChatML; DeepSeek-V3.x + V4 traces).\n\n## Mixture (fully shuffled)\n\n| source | rows | tokens |\n|---|---|---|\n")
    for s in sorted(tottok): f.write(f"| {s} | {totrows[s]:,} | {tottok[s]:,} |\n")
    f.write(f"| **TOTAL** | **{GR:,}** | **{GT:,}** |\n\n")
    f.write(f"{N} shards `data/train-XXXXX-of-{N:05d}.parquet` (zstd), row-level shuffled (seed {SEED}).\n")
    f.write("Budgets: proofs (math_proof, fineproofs, proofs_v2) all; math_notool 6B; math_withtool 10B; math_v4_cot/tir all.\n")
    f.write("Format: `<think> ` open, `\\n</think>\\n\\n` close; tool sources use Olmo `<function_calls>` pythonic. See INFERENCE_NOTES.md.\n")
print("DONE")
