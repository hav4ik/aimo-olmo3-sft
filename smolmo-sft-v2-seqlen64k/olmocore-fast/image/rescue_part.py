#!/usr/bin/env python3
"""Low-RAM rescue: turn a converter's partial .bin files into the final node-prefixed .npy,
WITHOUT the OOM-prone aggregate_stats. Reads .bin in 1GB chunks (peak RAM ~1.3GB).
Usage: rescue_part.py <tmp_dir with _*.partial.bin> <out_dir> <part_num>
Matches the converter's layout exactly: 1GB (268435456-token) uint32 chunks + bool mask + csv.gz."""
import sys, os, gzip, json, numpy as np
tmp, out, part = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(out, exist_ok=True)
tok_bin=os.path.join(tmp,"_tokens.partial.bin"); lab_bin=os.path.join(tmp,"_labels.partial.bin"); bnd_bin=os.path.join(tmp,"_boundaries.partial.bin")
for p in (tok_bin,lab_bin,bnd_bin):
    assert os.path.exists(p), f"missing {p}"
ITEM=4; CHUNK=(1*1024**3)//ITEM                      # 268435456 tokens / 1GB part (matches converter)
total=os.path.getsize(tok_bin)//ITEM
assert os.path.getsize(lab_bin)==total, f"labels({os.path.getsize(lab_bin)}) != tokens({total})"
tok=np.memmap(tok_bin,dtype=np.uint32,mode="r",shape=(total,))
lab=np.memmap(lab_bin,dtype=np.uint8,mode="r",shape=(total,))
bnd=np.memmap(bnd_bin,dtype=np.int64,mode="r"); bnd=np.asarray(bnd).reshape(-1,2)   # (start,end) per doc
maxid=0; trainable=0; nchunks=0
for ci,i in enumerate(range(0,total,CHUNK)):
    e=min(i+CHUNK,total)
    t=np.array(tok[i:e]); l=np.array(lab[i:e])        # ~1.25GB transient
    t.tofile(os.path.join(out,f"token_ids_part_{part:03d}_{ci:04d}.npy"))
    l.astype(np.bool_).tofile(os.path.join(out,f"labels_mask_part_{part:03d}_{ci:04d}.npy"))
    with gzip.open(os.path.join(out,f"token_ids_part_{part:03d}_{ci:04d}.csv.gz"),"wt") as f:
        for s,en in bnd:
            if en>i and s<e: f.write(f"{max(0,int(s)-i)},{min(e-i,int(en)-i)}\n")
    maxid=max(maxid,int(t.max())); trainable+=int(l.sum()); nchunks+=1
    del t,l
seq=bnd.shape[0]
json.dump({"configuration":{"max_sequence_length":65536,"tokenizer":"allenai/Olmo-3-7B-Think","chat_template":"olmo_thinker"},
  "overall_statistics":{"total_instances":seq,"total_tokens":total,"trainable_tokens":trainable,
  "trainable_percentage":trainable/total*100 if total else 0,"instances_filtered":0}},
  open(os.path.join(out,f"stats_part_{part:03d}.json"),"w"),indent=2)
print(f"rescued node {part}: {nchunks} parts, {seq:,} seq, {total:,} tokens, trainable {trainable/total*100:.2f}%, max_id {maxid}")
