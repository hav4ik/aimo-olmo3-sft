import numpy as np, gzip, glob
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/work/out_glob/tokenizer")
ids  = np.fromfile("/work/out_glob/token_ids_part_0000.npy", dtype=np.uint32)
mask = np.fromfile("/work/out_glob/labels_mask_part_0000.npy", dtype=np.bool_)
# document boundaries (start,end) per sequence
bounds=[]
for f in sorted(glob.glob("/work/out_glob/token_ids_part_*.csv.gz")):
    with gzip.open(f,"rt") as fh:
        for ln in fh:
            s,e=ln.strip().split(","); bounds.append((int(s),int(e)))
print(f"total tokens={len(ids):,}  sequences={len(bounds)}")
SPECIAL={100264:'<|im_start|>',100265:'<|im_end|>',100257:'<|endoftext|>'}
for di,(s,e) in enumerate(bounds[:2]):
    seq=ids[s:e]; m=mask[s:e]
    print(f"\n===== SEQUENCE {di}: {e-s} tokens, trained(mask=1)={int(m.sum())} masked(mask=0)={int((~m).sum())} =====")
    # walk role headers; show, for each <|im_start|>role, whether the header+content is masked or trained
    i=0
    while i < len(seq):
        if seq[i]==100264:  # <|im_start|>
            j=i+1
            while j<len(seq) and seq[j] not in (100265,100257): j+=1
            role_txt=tok.decode(seq[i+1:min(i+6,len(seq))]).split("\n")[0]
            span=m[i:j+1]
            frac=span.mean()
            tag = "TRAINED" if frac>0.5 else "masked "
            # show transition point inside assistant turns
            print(f"  turn '<|im_start|>{role_txt[:14]:<14}' tokens[{i}:{j+1}]  trained_frac={frac:.2f}  -> {tag}")
            i=j+1
        else:
            i+=1
    # show the exact mask transition at the first assistant turn: last 3 masked + first 4 trained tokens
    tr=np.where(m)[0]
    if len(tr):
        k=tr[0]
        ctx_ids=seq[max(0,k-4):k+4]; ctx_m=m[max(0,k-4):k+4]
        print("  first trained token at idx",k,"-> boundary context (tok|mask):")
        for t,b in zip(ctx_ids,ctx_m):
            print(f"     {repr(SPECIAL.get(int(t), tok.decode([int(t)]))):<16} mask={int(b)}")
