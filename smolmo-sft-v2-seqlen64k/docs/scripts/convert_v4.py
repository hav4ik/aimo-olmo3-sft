#!/usr/bin/env python3
"""Nemotron-SFT-Math-v4 -> 3 parquets (Olmo think format):
  math_v4_cot.parquet     : subset=cot (non-tool)
  math_v4_tir.parquet     : subset=tir WITH tool use (Olmo tool format, validated)
  math_v4_tir_nc.parquet  : subset=tir WITHOUT tool use -> NON-TOOL (gets non-tool system, no functions)
A conversation 'uses tools' iff it has a `tool` role message or an assistant `tool_calls`."""
import sys, json, hashlib, re
sys.path.insert(0,"/data/proof-redesign/scripts")
from cascade_to_olmo import OLMO_TOOL_SYSTEM, _olmo_parse
import pyarrow as pa, pyarrow.parquet as pq
IN="/data/Nemotron-SFT-Math-v4/data/train.jsonl"
COT="/data/proof-redesign/tables/math_v4_cot.parquet"
TIR="/data/proof-redesign/tables/math_v4_tir.parquet"
TNC="/data/proof-redesign/tables/math_v4_tir_nc.parquet"
def pid(p): return hashlib.sha256(re.sub(r"[^a-z0-9]","",(p or "").lower()).encode()).hexdigest()
def clean(s): return (s or "").replace("<think>","").replace("</think>","").strip()
M2=pa.struct([("role",pa.string()),("content",pa.string())])
M4=pa.struct([("role",pa.string()),("content",pa.string()),("functions",pa.string()),("function_calls",pa.string())])
SC=pa.schema([("messages",pa.list_(M2)),("task_type",pa.string()),("problem_id",pa.string())])
ST=pa.schema([("messages",pa.list_(M4)),("task_type",pa.string()),("problem_id",pa.string())])
def uses_tools(r):
    ms=r.get("messages",[])
    return any(m.get("role")=="tool" for m in ms) or any(m.get("role")=="assistant" and m.get("tool_calls") for m in ms)
def conv_cot(r):  # non-tool: first user + first assistant
    us=[m for m in r["messages"] if m.get("role")=="user"]; aa=[m for m in r["messages"] if m.get("role")=="assistant"]
    if not us or not aa: return None
    rc=clean(aa[0].get("reasoning_content")); c=clean(aa[0].get("content"))
    if not rc or not c: return None
    return [{"role":"user","content":us[0].get("content")},{"role":"assistant","content":f"<think> {rc}\n</think>\n\n{c}"}]
def conv_tir(r):
    tools=r.get("tools"); funcs=json.dumps(json.loads(tools) if isinstance(tools,str) else tools)
    out=[{"role":"system","content":OLMO_TOOL_SYSTEM,"functions":funcs,"function_calls":None}]
    for m in r["messages"]:
        role=m.get("role")
        if role=="user": out.append({"role":"user","content":m.get("content"),"functions":None,"function_calls":None})
        elif role=="tool": out.append({"role":"environment","content":m.get("content"),"functions":None,"function_calls":None})
        elif role=="assistant":
            rc=clean(m.get("reasoning_content")); c=clean(m.get("content")); tcs=m.get("tool_calls")
            content=f"<think> {rc}\n</think>"+(f"\n\n{c}" if c else "")
            fc=None
            if tcs:
                lines=[]; orig=[]
                for tc in tcs:
                    fn=tc.get("function",{}); name=fn.get("name"); args=fn.get("arguments")
                    if isinstance(args,str): args=json.loads(args)
                    lines.append(f"{name}(" + ", ".join(f"{k}={v!r}" for k,v in args.items()) + ")"); orig.append((name,args))
                fc="\n".join(lines)
                if _olmo_parse(fc)!=orig: return None
            out.append({"role":"assistant","content":content,"functions":None,"function_calls":fc})
        else: return None
    return out
wc=pq.ParquetWriter(COT,SC,compression="zstd"); wt=pq.ParquetWriter(TIR,ST,compression="zstd"); wn=pq.ParquetWriter(TNC,SC,compression="zstd")
cb=[];tb=[];nb=[]; ncot=ntir=ntnc=skip=quar=0
with open(IN) as fh:
    for line in fh:
        line=line.strip()
        if not line: continue
        try: r=json.loads(line)
        except: skip+=1; continue
        sub=r.get("subset"); pi=pid(r.get("problem"))
        if sub=="cot":
            m=conv_cot(r)
            if m is None: skip+=1; continue
            cb.append({"messages":m,"task_type":"cot","problem_id":pi}); ncot+=1
        elif sub=="tir":
            if uses_tools(r):
                m=conv_tir(r)
                if m is None: quar+=1; continue
                tb.append({"messages":m,"task_type":"tir","problem_id":pi}); ntir+=1
            else:
                m=conv_cot(r)
                if m is None: skip+=1; continue
                nb.append({"messages":m,"task_type":"tir_nocall","problem_id":pi}); ntnc+=1
        else: skip+=1; continue
        if len(cb)>=1000: wc.write_table(pa.Table.from_pylist(cb,schema=SC)); cb=[]
        if len(tb)>=500: wt.write_table(pa.Table.from_pylist(tb,schema=ST)); tb=[]
        if len(nb)>=1000: wn.write_table(pa.Table.from_pylist(nb,schema=SC)); nb=[]
        if (ncot+ntir+ntnc)%50000==0: print(f"  cot={ncot:,} tir={ntir:,} tir_nc={ntnc:,} quar={quar}",flush=True)
if cb: wc.write_table(pa.Table.from_pylist(cb,schema=SC))
if tb: wt.write_table(pa.Table.from_pylist(tb,schema=ST))
if nb: wn.write_table(pa.Table.from_pylist(nb,schema=SC))
wc.close(); wt.close(); wn.close()
print(f"DONE cot={ncot:,} tir(tool)={ntir:,} tir_nc(nontool)={ntnc:,} skipped={skip:,} quar={quar}")
