import sys, os, json, pyarrow.parquet as pq
sys.path.insert(0,"/data/proof-redesign/scripts")
import build_mix as B
from multiprocessing import Pool
os.makedirs(B.PARTS, exist_ok=True)
def run(skey, pool):
    p=B.SRC[skey]["path"]; nrg=pq.ParquetFile(p).metadata.num_row_groups
    print(f"{skey}: {nrg} row-groups (take all)",flush=True)
    a=dict(kept=0,tokens=0,total=0,nothink=0,long=0); done=0
    for r in pool.imap_unordered(B.worker, [(skey,p,rg) for rg in range(nrg)]):
        a["kept"]+=r["n_kept"]; a["tokens"]+=r["sum_tokens"]; a["total"]+=r["n_total"]
        a["nothink"]+=r["n_nothink"]; a["long"]+=r["n_long"]; done+=1
        if done%50==0: print(f"  {skey}: {done}/{nrg} rgs, {a['tokens']:,} tok",flush=True)
    print(f"{skey} DONE: kept={a['kept']:,} tokens={a['tokens']:,} (drop nothink={a['nothink']:,} long={a['long']:,})",flush=True)
    return a
with Pool(B.N, initializer=B.init_worker) as pool:
    res={s:run(s,pool) for s in ["math_v4_tir","math_v4_tir_nc"]}
print("TIR-SPLIT DONE:", json.dumps({s:{"kept":a["kept"],"tokens":a["tokens"]} for s,a in res.items()}))
