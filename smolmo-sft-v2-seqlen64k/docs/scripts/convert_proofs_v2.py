#!/usr/bin/env python3
"""Nemotron-Math-Proofs-v2 train.jsonl -> tables/proofs_v2.parquet (Olmo think format).
assistant.content = "<think> " + reasoning_content + "\n</think>\n\n" + content (the real V4 trace)."""
import json, hashlib, re, pyarrow as pa, pyarrow.parquet as pq
IN ="/data/Nemotron-Math-Proofs-v2/data/train.jsonl"
OUT="/data/proof-redesign/tables/proofs_v2.parquet"
def ptype(u):
    if "Your task is to solve" in u: return "solution"
    if "Your task is to evaluate" in u: return "evaluation"
    if "assess the whether" in u: return "analysis"
    return "other"
def pid(p): return hashlib.sha256(re.sub(r"[^a-z0-9]","",(p or "").lower()).encode()).hexdigest()
MSG=pa.struct([("role",pa.string()),("content",pa.string())])
SCH=pa.schema([("messages",pa.list_(MSG)),("task_type",pa.string()),("problem_id",pa.string())])
w=pq.ParquetWriter(OUT, SCH, compression="zstd")
buf=[]; n=skip=0; types={}
with open(IN) as fh:
    for line in fh:
        line=line.strip()
        if not line: continue
        try: r=json.loads(line)
        except: skip+=1; continue
        ms=r.get("messages",[])
        us=[m for m in ms if m.get("role")=="user"]; as_=[m for m in ms if m.get("role")=="assistant"]
        if not us or not as_: skip+=1; continue
        u=us[0].get("content") or ""; a=as_[0]
        rc=a.get("reasoning_content"); c=a.get("content")
        if not rc or not c: skip+=1; continue
        rc=rc.replace("<think>","").replace("</think>","")   # defensive: only our inserted tags
        c=c.replace("<think>","").replace("</think>","")
        asst="<think> "+rc.strip()+"\n</think>\n\n"+c.strip()
        t=ptype(u); types[t]=types.get(t,0)+1
        buf.append({"messages":[{"role":"user","content":u},{"role":"assistant","content":asst}],
                    "task_type":t,"problem_id":pid(r.get("problem"))})
        n+=1
        if len(buf)>=1000:
            w.write_table(pa.Table.from_pylist(buf,schema=SCH)); buf=[]
            if n % 20000==0: print(f"  {n:,} converted...",flush=True)
if buf: w.write_table(pa.Table.from_pylist(buf,schema=SCH))
w.close()
print(f"DONE converted={n:,} skipped={skip:,} types={types} -> {OUT}")
