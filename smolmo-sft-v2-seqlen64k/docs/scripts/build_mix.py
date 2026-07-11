#!/usr/bin/env python3
"""Build the final v2 SFT mix (Olmo-3 format).

Sources -> filters (drop no-<think>, drop rendered length >= MAXLEN) -> unified sharded parquet.
  math_proof  = solutions + assessment        : take ALL
  fineproofs  = FineProofs-SFT raw            : take ALL
  math_notool = math_notool.parquet           : stream (shuffled row-groups) until 12B tokens
  math_withtool = math_tool_olmo.parquet      : stream (shuffled row-groups) until 12B tokens

num_tokens = full rendered length (system + special tokens + everything), via the Olmo template.
Output: OUT/data/train-XXXXX-of-NNNNN.parquet (zstd) + MIXTURE.md + stats.json.
"""
import os, sys, glob, json, math, random, time, traceback
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import pyarrow as pa, pyarrow.parquet as pq

TOK_DIR  = "/data/aimo-proof-pilot/smolmo-proofs-cot-sft/olmocore/tokenizer"
OUT      = "/data/proof-redesign/smolmo-math-cot-sft"
PARTS    = os.path.join(OUT, "_parts")
DATA     = os.path.join(OUT, "data")
MAXLEN   = 65536
BUDGET   = 12_000_000_000
SEED     = 1234
N        = int(os.environ.get("NWORKERS", min((os.cpu_count() or 8) - 2, 12)))

NONTOOL  = "You are an expert mathematical assistant. Provide rigorous, complete proofs. You are not allowed to use tools."

SRC = {  # path, label, kind(nontool|tool|fineproofs), columns, budget(None=all)
  "solutions":  dict(path="/data/proof-redesign/tables/solutions.parquet",  label="math_proof",  kind="nontool",
                     cols=["messages","generator","source","problem_id","domain"], budget=None),
  "assessment": dict(path="/data/proof-redesign/tables/assessment.parquet", label="math_proof",  kind="nontool",
                     cols=["messages","generator","source","problem_id","domain","task_type","score_normalized"], budget=None),
  "fineproofs": dict(path=None,  label="fineproofs", kind="fineproofs",
                     cols=["messages","source"], budget=None),  # multi-file, handled specially
  "notool":     dict(path="/data/Nemotron-Cascade-2-SFT-Data/math/math_notool.parquet", label="math_notool", kind="nontool",
                     cols=["messages","generator","source","domain"], budget=BUDGET),
  "tool":       dict(path="/data/proof-redesign/tables/math_tool_olmo.parquet", label="math_withtool", kind="tool",
                     cols=["messages","generator","source","domain"], budget=BUDGET),
  "proofs_v2":  dict(path="/data/proof-redesign/tables/proofs_v2.parquet", label="proofs_v2", kind="nontool",
                     cols=["messages","task_type","problem_id"], budget=None),  # DeepSeek-V4 proofs; reasoning_content already assembled into <think>
  "math_v4_cot":dict(path="/data/proof-redesign/tables/math_v4_cot.parquet", label="math_v4_cot", kind="nontool",
                     cols=["messages","task_type","problem_id"], budget=None),  # DeepSeek-V4 cot (numeric)
  "math_v4_tir":dict(path="/data/proof-redesign/tables/math_v4_tir.parquet", label="math_v4_tir", kind="tool",
                     cols=["messages","task_type","problem_id"], budget=None),  # DeepSeek-V4 TIR that ACTUALLY calls tools (Olmo tool format)
  "math_v4_tir_nc":dict(path="/data/proof-redesign/tables/math_v4_tir_nc.parquet", label="math_v4_tir_nc", kind="nontool",
                     cols=["messages","task_type","problem_id"], budget=None),  # DeepSeek-V4 TIR with NO tool call -> non-tool system
}

MSG = pa.struct([("role",pa.string()),("content",pa.string()),("functions",pa.string()),("function_calls",pa.string())])
SCHEMA = pa.schema([("messages",pa.list_(MSG)),("source",pa.string()),("num_tokens",pa.int32()),
   ("task_type",pa.string()),("score_normalized",pa.float64()),("generator",pa.string()),
   ("orig_source",pa.string()),("problem_id",pa.string()),("domain",pa.string())])

_tok = None
def init_worker():
    global _tok
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(TOK_DIR)

def has_think(conv):
    a = [m for m in conv if m.get("role") == "assistant"]
    if not a: return False
    for m in a:
        c = m.get("content") or ""
        if not c.lstrip().startswith("<think>"): return False
        i, j = c.find("<think>"), c.find("</think>")
        if i < 0 or j < 0 or j < i: return False
    return True

def fix_think(role, c):
    # Standardize assistant reasoning markers (see INFERENCE_NOTES.md):
    #  OPEN  -> "<think> {reasoning}" (SPACE, Dolma-style): makes the stock Olmo gen-prompt
    #           "<think>" (tok 29) an exact token-prefix of training -> no skew, clean mask.
    #  CLOSE -> "{reasoning}\n</think>\n\n{answer}" for answer turns (Dolci Think SFT form);
    #           "{reasoning}\n</think>" for tool-call turns (no answer text; <function_calls> next).
    if role != "assistant" or not c:
        return c
    if c.startswith("<think>"):
        c = "<think> " + c[len("<think>"):].lstrip()
    k = c.find("</think>")
    if k >= 0:
        before = c[:k].rstrip()
        after = c[k + len("</think>"):].lstrip()
        c = before + "\n</think>" + ("\n\n" + after if after else "")
    return c

def bake_nontool(conv):
    out = [{"role":"system","content":NONTOOL,"functions":None,"function_calls":None}]
    for m in conv:
        if m.get("role") == "system": continue
        r = m.get("role")
        out.append({"role":r,"content":fix_think(r, m.get("content")),"functions":None,"function_calls":None})
    return out

def keep_tool(conv):
    return [{"role":m.get("role"),"content":fix_think(m.get("role"), m.get("content")),
             "functions":m.get("functions"),"function_calls":m.get("function_calls")} for m in conv]

def record(skey, row, msgs, ntok):
    lab = SRC[skey]["label"]
    tt = {"solutions":"solution","fineproofs":"solution","notool":"notool","tool":"tool"}.get(skey) or row.get("task_type")
    V4={"math_v4_cot","math_v4_tir","math_v4_tir_nc"}
    gen = "DeepSeek-V4" if (skey in V4 or skey=="proofs_v2") else (None if skey=="fineproofs" else row.get("generator"))
    osrc = {"proofs_v2":"Nemotron-Math-Proofs-v2"}.get(skey, "Nemotron-SFT-Math-v4" if skey in V4 else row.get("source"))
    dom = {"fineproofs":"olympiad","proofs_v2":"proof","math_v4_cot":"math","math_v4_tir":"math_tool","math_v4_tir_nc":"math"}.get(skey, row.get("domain"))
    pid = row.get("problem_id") if skey in ("solutions","assessment","proofs_v2","math_v4_cot","math_v4_tir","math_v4_tir_nc") else None
    return dict(messages=msgs, source=lab, num_tokens=int(ntok), task_type=tt,
        score_normalized=(row.get("score_normalized") if skey=="assessment" else None),
        generator=gen, orig_source=osrc, problem_id=pid, domain=dom)

CHUNK = 128       # tokenize in small batches: keep only lengths, never hold a whole row-group of ids
CAP_CHARS = 400_000  # prefix cap: avoid full BPE on monster reasoning traces. A kept row (<MAXLEN tok)
                     # is always < this many chars; truncation caps the output so overlong detection is exact.
def process_rows(skey, rows):
    """rows: list of pylist dicts. Returns (records, n_total, n_nothink, n_long). Exact num_tokens."""
    kind = SRC[skey]["kind"]
    n_total = len(rows); n_nothink = n_long = 0; recs = []
    buf = []  # (row, baked) pending tokenization
    def lens_capped(txts):
        enc = _tok([t[:CAP_CHARS] for t in txts], add_special_tokens=False, truncation=True, max_length=MAXLEN)
        return [len(x) for x in enc["input_ids"]]
    def flush():
        nonlocal n_long
        if not buf: return
        texts = [_tok.apply_chat_template(b, tokenize=False, add_generation_prompt=False) for _, b in buf]
        recheck = []
        for (row, baked), L, t in zip(buf, lens_capped(texts), texts):
            if L >= MAXLEN: n_long += 1                       # prefix already >=MAXLEN -> overlong (exact)
            elif len(t) <= CAP_CHARS: recs.append(record(skey, row, baked, L))  # prefix==full -> exact
            else: recheck.append((row, baked, t))             # long-but-sparse: need exact full pass (rare)
        if recheck:                                           # exact fallback for the rare case
            enc = _tok([t for _,_,t in recheck], add_special_tokens=False, truncation=True, max_length=MAXLEN)
            for (row, baked, _), ids in zip(recheck, enc["input_ids"]):
                if len(ids) >= MAXLEN: n_long += 1
                else: recs.append(record(skey, row, baked, len(ids)))
        buf.clear()
    for row in rows:
        conv = row["messages"]
        if not has_think(conv): n_nothink += 1; continue
        buf.append((row, keep_tool(conv) if kind == "tool" else bake_nontool(conv)))
        if len(buf) >= CHUNK: flush()
    flush()
    return recs, n_total, n_nothink, n_long

def worker(task):
    skey, path, rg = task
    try:
        cols = SRC[skey]["cols"]
        rows = pq.ParquetFile(path).read_row_group(rg, columns=cols).to_pylist()
        recs, nt, nn, nl = process_rows(skey, rows)
        if not recs:
            return dict(skey=skey, rg=rg, path=None, n_total=nt, n_nothink=nn, n_long=nl, n_kept=0, sum_tokens=0)
        stem = os.path.splitext(os.path.basename(path))[0]  # unique per file (fineproofs has 2 files w/ same rg idx)
        outp = os.path.join(PARTS, f"{skey}_{stem}_{rg:05d}.parquet")
        pq.write_table(pa.Table.from_pylist(recs, schema=SCHEMA), outp, compression="zstd")
        return dict(skey=skey, rg=rg, path=outp, n_total=nt, n_nothink=nn, n_long=nl,
                    n_kept=len(recs), sum_tokens=int(sum(r["num_tokens"] for r in recs)))
    except Exception as e:
        sys.stderr.write(f"[worker err {skey} rg{rg}] {e}\n{traceback.format_exc()}\n"); sys.stderr.flush()
        return dict(skey=skey, rg=rg, path=None, n_total=0, n_nothink=0, n_long=0, n_kept=0, sum_tokens=0, error=str(e))

def fineproofs_tasks():
    files = sorted(glob.glob("/data/FineProofs-SFT/data/*.parquet"))
    return [("fineproofs", f, rg) for f in files for rg in range(pq.ParquetFile(f).metadata.num_row_groups)]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def main():
    os.makedirs(PARTS, exist_ok=True); os.makedirs(DATA, exist_ok=True)
    from multiprocessing import Pool
    log(f"N workers={N}  MAXLEN={MAXLEN}  BUDGET={BUDGET:,}")
    results = []  # in final shard order
    agg = {}      # per-label stats
    def add(res_list):
        for r in res_list:
            lab = SRC[r["skey"]]["label"]
            a = agg.setdefault(lab, dict(kept=0,tokens=0,total=0,nothink=0,long=0))
            a["kept"]+=r["n_kept"]; a["tokens"]+=r["sum_tokens"]; a["total"]+=r["n_total"]
            a["nothink"]+=r["n_nothink"]; a["long"]+=r["n_long"]
            if r["path"]: results.append(r)

    with Pool(N, initializer=init_worker) as pool:
        # --- take-all sources ---
        for skey in ["solutions","assessment"]:
            p = SRC[skey]["path"]; nrg = pq.ParquetFile(p).metadata.num_row_groups
            log(f"{skey}: {nrg} row-groups (take all)")
            add(pool.map(worker, [(skey,p,rg) for rg in range(nrg)]))
            log(f"  {skey} done: kept={agg[SRC[skey]['label']]['kept']:,} tokens={agg[SRC[skey]['label']]['tokens']:,}")
        # fineproofs (multi-file)
        ftasks = fineproofs_tasks(); log(f"fineproofs: {len(ftasks)} row-groups (take all)")
        # worker reads by (skey,path,rg) — fineproofs path varies, so call directly
        add(pool.map(worker, ftasks))
        log(f"  fineproofs done: kept={agg['fineproofs']['kept']:,} tokens={agg['fineproofs']['tokens']:,}")

        # --- budgeted sources (adaptive waves over shuffled row-groups) ---
        for skey in ["notool","tool"]:
            p = SRC[skey]["path"]; nrg = pq.ParquetFile(p).metadata.num_row_groups
            order = list(range(nrg)); random.Random(SEED).shuffle(order)
            log(f"{skey}: {nrg} row-groups, budget {BUDGET:,} (shuffled stream)")
            total = 0; done = 0; i = 0
            while total < BUDGET and i < len(order):
                if done == 0:
                    wave = min(N, len(order))
                else:
                    tpr = total / done
                    need = max(1, math.ceil((BUDGET - total) / max(tpr, 1)))
                    wave = min(need, 2 * N, len(order) - i)
                batch = [(skey, p, rg) for rg in order[i:i+wave]]; i += wave
                res = pool.map(worker, batch); add(res)
                total += sum(r["sum_tokens"] for r in res); done += len(res)
                log(f"  {skey}: {done} rg, {total:,} tok ({total/BUDGET*100:.1f}% of budget)")
            log(f"  {skey} done: kept={agg[SRC[skey]['label']]['kept']:,} tokens={agg[SRC[skey]['label']]['tokens']:,}")

    # --- finalize: rename parts to HF-convention sharded files ---
    Ntot = len(results)
    log(f"finalizing {Ntot} shards -> {DATA}")
    for k, r in enumerate(results):
        os.rename(r["path"], os.path.join(DATA, f"train-{k:05d}-of-{Ntot:05d}.parquet"))
    try: os.rmdir(PARTS)
    except OSError: pass

    grand = dict(rows=sum(a["kept"] for a in agg.values()), tokens=sum(a["tokens"] for a in agg.values()),
                 shards=Ntot, per_source=agg)
    # Top-level layout mirrors chankhavu/smolmo-proofs-cot-sft: README.md + data/ (olmocore/ added later).
    with open(os.path.join(OUT,"README.md"),"w") as f:
        f.write("# smolmo-math-cot-sft\n\n")
        f.write("Math chain-of-thought SFT mixture for Olmo-3.1-Think, in the model's native ChatML/`<think>` "
                "format (verified against `allenai/Olmo-3-7B-Think`). Three task families:\n"
                "- **math_proof** — proof writing + rubric/grading assessment (Nemotron-Cascade-2 `math_proof`).\n"
                "- **math_notool** — numeric-answer reasoning, no tools (Nemotron-Cascade-2 `math_notool`).\n"
                "- **math_withtool** — tool-augmented reasoning with a stateful Python interpreter, Olmo native "
                "`<function_calls>` format (Nemotron-Cascade-2 `math_tool`, converted).\n"
                "- **fineproofs** — high-quality olympiad proofs (FineProofs-SFT).\n\n")
        f.write("## Mixture\n\n")
        f.write("| source | rows | tokens | scanned | dropped: no-`<think>` | dropped: ≥65536 tok |\n|---|---|---|---|---|---|\n")
        for lab,a in agg.items():
            f.write(f"| {lab} | {a['kept']:,} | {a['tokens']:,} | {a['total']:,} | {a['nothink']:,} | {a['long']:,} |\n")
        f.write(f"| **TOTAL** | **{grand['rows']:,}** | **{grand['tokens']:,}** | | | |\n\n")
        f.write("Budgets: math_proof + fineproofs taken in full; math_notool and math_withtool each streamed "
                f"(shuffled row-groups, seed {SEED}) to ~{BUDGET//10**9}B tokens.\n\n")
        f.write("## Filters\n"
                f"- Drop any sample whose assistant turn(s) lack a well-formed `<think>…</think>`.\n"
                f"- Drop any sample whose full rendered length ≥ {MAXLEN} tokens.\n"
                "- `num_tokens` = full Olmo-template render (system + special tokens + every turn), exact.\n\n")
        f.write("## Schema (`data/*.parquet`)\n`" + "`, `".join(SCHEMA.names) + "`\n"
                "- `messages`: list of `{role, content, functions, function_calls}` (Olmo native; "
                "`functions`/`function_calls` null on non-tool rows).\n"
                "- `source` ∈ {math_proof, math_notool, math_withtool, fineproofs}; `task_type` ∈ "
                "{solution, evaluation, analysis, notool, tool}.\n\n")
        f.write("## Format\n"
                "- Reasoning opens with `<think> ` (space) and closes with `\\n</think>\\n\\n{answer}` "
                "(Dolci-aligned); tool-call turns close with `\\n</think>` then `<function_calls>`.\n"
                "- System prompts: non-tool sources → `"+NONTOOL+"`; math_withtool → tool prompt + a "
                "`functions` schema. Final assistant turn ends with `<|endoftext|>`.\n"
                "- Serve with the upstream `allenai/Olmo-3-7B-Think` chat template; vLLM "
                "`--reasoning-parser olmo3 --tool-call-parser olmo3`. See `INFERENCE_NOTES.md` for the "
                "`<think> ` token-prefix rationale and masking.\n")
    log(f"DONE. rows={grand['rows']:,} tokens={grand['tokens']:,} shards={Ntot} -> {OUT}/{{README.md,data/}}")
    log("per-source: " + json.dumps(agg))

if __name__ == "__main__":
    main()
