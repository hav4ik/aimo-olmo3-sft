import glob, pyarrow.parquet as pq, pandas as pd, numpy as np, seaborn as sns, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
shards=sorted(glob.glob("/mnt/data/proof-redesign/smolmo-math-cot-sft/data/*.parquet"))
src=[]; tok=[]
for p in shards:
    t=pq.read_table(p, columns=["source","num_tokens"])
    src += t.column("source").to_pylist(); tok += t.column("num_tokens").to_pylist()
df=pd.DataFrame({"source":src,"num_tokens":np.array(tok,dtype=np.int32)})
print("rows:",len(df))
ORDER=["math_proof","fineproofs","proofs_v2","math_notool","math_v4_cot","math_v4_tir_nc","math_withtool","math_v4_tir"]
GROUP={"math_proof":"proofs","fineproofs":"proofs","proofs_v2":"proofs","math_notool":"no-tool","math_v4_cot":"no-tool","math_v4_tir_nc":"no-tool","math_withtool":"tool","math_v4_tir":"tool"}
pal={"proofs":"#4C72B0","no-tool":"#55A868","tool":"#C44E52"}
# ---- stats table ----
def q(x,p): return int(np.percentile(x,p))
rows=[]
for s in ORDER:
    x=df[df.source==s].num_tokens.values
    rows.append([s,GROUP[s],len(x),int(x.mean()),int(np.median(x)),q(x,90),q(x,99),int(x.max()),round(x.sum()/1e9,2)])
st=pd.DataFrame(rows,columns=["source","group","rows","mean","median","p90","p99","max","tokens_B"])
st.to_csv("/mnt/data/proof-redesign/smolmo-math-cot-sft/assets/token_stats.csv",index=False)
print(st.to_string(index=False))
# ---- Figure A: faceted histograms ----
sns.set_theme(style="whitegrid", context="talk")
g=sns.displot(df, x="num_tokens", col="source", col_order=ORDER, col_wrap=4, bins=45,
              height=3.0, aspect=1.25, facet_kws=dict(sharey=False),
              hue="source", hue_order=ORDER, palette=[pal[GROUP[s]] for s in ORDER], legend=False)
g.set_titles("{col_name}")
for ax,s in zip(g.axes.flat, ORDER):
    m=np.median(df[df.source==s].num_tokens)
    ax.axvline(m,color="k",ls="--",lw=1); ax.set_xlim(0,65536)
    ax.set_title(f"{s}\n(med {int(m):,} tok, n={ (df.source==s).sum():,})", fontsize=11)
    ax.set_xlabel("tokens"); ax.tick_params(labelsize=8)
g.fig.suptitle("smolmo-sft-v2-seqlen64k — token-length distribution per source (cap 65,536)", y=1.02, fontsize=14)
g.savefig("/mnt/data/proof-redesign/smolmo-math-cot-sft/assets/token_length_by_source.png", dpi=130, bbox_inches="tight")
plt.close("all")
# ---- Figure B: box + ECDF comparison ----
fig,(a1,a2)=plt.subplots(1,2,figsize=(16,6))
sns.boxenplot(data=df,y="source",x="num_tokens",order=ORDER,ax=a1,
              palette=[pal[GROUP[s]] for s in ORDER], hue="source", hue_order=ORDER, legend=False)
a1.set_xlim(0,65536); a1.set_title("Token length by source (boxen)"); a1.set_xlabel("tokens"); a1.set_ylabel("")
for s in ORDER:
    x=np.sort(df[df.source==s].num_tokens.values); y=np.arange(1,len(x)+1)/len(x)
    a2.plot(x,y,label=s,color=pal[GROUP[s]],alpha=0.9,lw=2)
a2.set_xlim(0,65536); a2.set_title("ECDF by source"); a2.set_xlabel("tokens"); a2.set_ylabel("fraction ≤ x")
a2.legend(fontsize=8,loc="lower right"); a2.grid(True,alpha=0.3)
fig.suptitle("smolmo-sft-v2-seqlen64k — token-length comparison",fontsize=14)
fig.tight_layout(); fig.savefig("/mnt/data/proof-redesign/smolmo-math-cot-sft/assets/token_length_compare.png",dpi=130,bbox_inches="tight")
print("saved figures + stats")
