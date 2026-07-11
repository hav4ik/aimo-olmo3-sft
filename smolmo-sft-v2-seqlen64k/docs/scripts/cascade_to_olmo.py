#!/usr/bin/env python3
"""Convert Nemotron-Cascade-2 tool-format messages → Olmo-3 native tool format, with a
validation gate that QUARANTINES any row that wouldn't round-trip / parse faithfully.

Cascade has TWO tool-result encodings (both handled):
  (a) role="user",  content = "<tool_response>\\nOUT\\n</tool_response>"
  (b) role="tool",  content = "<|im_start|>user\\n<tool_response>\\nOUT\\n</tool_response>\\n"  (double-wrapped)
Both → Olmo role="environment", content=OUT. The real problem `user` (no <tool_response>) stays `user`.

Olmo-3 native (chat_template.jinja + allenai/Dolci-*-Tool-Use + vLLM Olmo3PythonicToolParser):
  - system: canonical content + `functions` = json.dumps(OpenAI tool array)
  - assistant call turn: `content`=reasoning(<think>…</think>); `function_calls`=PYTHONIC newline-joined
    `name(arg=<py-literal>)` (one call per physical line — parser splits on '\\n')
  - tool result: role="environment", raw output

VALIDATION GATE (run per row in the CLI): for every assistant turn, the emitted `function_calls`
is parsed with the EXACT Olmo3PythonicToolParser logic — the real TOOL_CALL_REGEX (regex module,
1 s timeout, so ReDoS-prone calls are caught) + ast + literal_eval — and byte-compared to the
original code; tag-balance is checked to catch values containing </parameter>/</function></tool_call>
(silent truncation); and tool-result→environment mapping is verified. Failing rows are quarantined.
"""
import re, ast, json
try:
    import regex as _rx           # supports match(..., timeout=)
    _HAS_RX = True
except Exception:
    _rx = re; _HAS_RX = False

OLMO_TOOL_SYSTEM = (
    "You are an expert mathematical assistant. Provide rigorous, complete solutions. "
    "You are provided with function signatures within <functions></functions> XML tags. "
    "You may call one or more functions to assist with the user query. Output any function "
    "calls within <function_calls></function_calls> XML tags. Don't make assumptions about "
    "what values to plug into functions."
)
VALID_ROLES = {"system", "user", "assistant", "environment"}

_TOOLS  = re.compile(r"<tools>(.*?)</tools>", re.S)
_FUNC   = re.compile(r"<function>(.*?)</function>", re.S)
_NAME   = re.compile(r"<name>(.*?)</name>", re.S)
_DESC   = re.compile(r"<description>(.*?)</description>", re.S)
_PARAMS = re.compile(r"<parameters>(.*?)</parameters>", re.S)
_PARAM  = re.compile(r"<parameter>(.*?)</parameter>", re.S)
_TYPE   = re.compile(r"<type>(.*?)</type>", re.S)
_REQ    = re.compile(r"<required>(.*?)</required>", re.S)
_TCALL  = re.compile(r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>", re.S)
_CALLP  = re.compile(r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>", re.S)
_TRESP  = re.compile(r"<tool_response>\n?(.*)\n?</tool_response>", re.S)   # GREEDY: to last close tag
_TR_START = re.compile(r"^\s*(?:<\|im_start\|>[^\n]*\n)?<tool_response>")  # boundary-anchored

# Exact regex copied from vLLM Olmo3PythonicToolParser.TOOL_CALL_REGEX
_OLMO_TOOL_CALL_REGEX = _rx.compile(
    r"\[([a-zA-Z]+\w*\(([a-zA-Z]+\w*=.*,\s*)*([a-zA-Z]+\w*=.*\s)?\),\s*)*"
    r"([a-zA-Z]+\w*\(([a-zA-Z]+\w*=.*,\s*)*([a-zA-Z]+\w*=.*\s*)?\)\s*)+\]",
    _rx.DOTALL,
)
_SENTINELS = ("</parameter>", "<parameter=", "</function>", "<function=", "<tool_call>", "</tool_call>")


def parse_tools(system_content):
    mt = _TOOLS.search(system_content)
    if not mt:
        return None
    tools = []
    for fn in _FUNC.finditer(mt.group(1)):
        body = fn.group(1)
        nm, ds = _NAME.search(body), _DESC.search(body)
        params = {"type": "object", "properties": {}, "required": []}
        pm = _PARAMS.search(body)
        if pm:
            for p in _PARAM.finditer(pm.group(1)):
                pb = p.group(1)
                pn, pt, pd = _NAME.search(pb), _TYPE.search(pb), _DESC.search(pb)
                if pn:
                    prop = {"type": (pt.group(1).strip() if pt else "string")}
                    if pd:
                        prop["description"] = pd.group(1).strip()
                    params["properties"][pn.group(1).strip()] = prop
            rq = _REQ.search(pm.group(1))
            if rq:
                try:
                    params["required"] = json.loads(rq.group(1).strip())
                except Exception:
                    params["required"] = [x.strip().strip('"\'') for x in rq.group(1).strip("[] \n").split(",") if x.strip()]
        tools.append({"type": "function", "function": {
            "name": nm.group(1).strip() if nm else "",
            "description": ds.group(1).strip() if ds else "",
            "parameters": params}})
    return json.dumps(tools)


def is_tool_result(content):
    return bool(_TR_START.match(content))


def strip_tool_response(content):
    # Strip the leading framing newline (unambiguous wrapper) via `\n?`; KEEP any trailing newline:
    # environment content is masked context (not a generation target), and Cascade's trailing \n is
    # plausibly the tool's real output newline (e.g. print()), so preserving the bytes is faithful.
    m = _TRESP.search(content)
    return m.group(1) if m else content


def _calls_from_assistant(content):
    """[(name, [(param, value_str), ...]), ...] using the Cascade XML, in order."""
    calls = []
    for tc in _TCALL.finditer(content):
        params = [(pm.group(1).strip(), pm.group(2)) for pm in _CALLP.finditer(tc.group(2))]
        calls.append((tc.group(1).strip(), params))
    return calls


def parse_assistant(content):
    """-> (content_before_first_call, function_calls_str_or_None)."""
    matches = list(_TCALL.finditer(content))
    if not matches:
        return content, None
    pre = content[:matches[0].start()].rstrip()
    lines = []
    for name, params in _calls_from_assistant(content):
        args = ", ".join(f"{p}={v!r}" for p, v in params)
        lines.append(f"{name}({args})")
    return pre, "\n".join(lines)


def convert_messages(messages):
    out = []
    for m in messages:
        role = m["role"]
        content = m.get("content") or ""
        if is_tool_result(content):                       # role user OR tool, double-wrapped or not
            out.append({"role": "environment", "content": strip_tool_response(content)})
        elif role == "system":
            fns = parse_tools(content)
            out.append({"role": "system", "content": OLMO_TOOL_SYSTEM, "functions": fns}
                       if fns is not None else {"role": "system", "content": content})
        elif role == "assistant":
            pre, fc = parse_assistant(content)
            msg = {"role": "assistant", "content": pre}
            if fc is not None:
                msg["function_calls"] = fc
            out.append(msg)
        else:                                             # problem user (or any other) → user/passthrough
            out.append({"role": ("user" if role in ("user", "tool") else role), "content": content})
    return out


# ----------------------------- validation gate -----------------------------
def _olmo_parse(function_calls, timeout=1.0):
    """Replicate Olmo3PythonicToolParser.extract_tool_calls; return [(name,{param:val})] or None
    (None = parser would reject / ReDoS timeout)."""
    if not function_calls or not function_calls.strip():
        return []
    body = ", ".join(l.strip() for l in function_calls.strip().splitlines() if l.strip())
    wrapped = f"[{body}]"
    try:
        m = _OLMO_TOOL_CALL_REGEX.match(wrapped, timeout=timeout) if _HAS_RX else _OLMO_TOOL_CALL_REGEX.match(wrapped)
    except TimeoutError:
        return None
    if m is None:
        return None
    try:
        lst = ast.parse(wrapped).body[0].value
        if not (isinstance(lst, ast.List) and all(isinstance(e, ast.Call) for e in lst.elts)):
            return None
        return [(e.func.id, {k.arg: ast.literal_eval(k.value) for k in e.keywords}) for e in lst.elts]
    except Exception:
        return None


def validate(cascade_msgs):
    """(ok, reason). Quarantine criteria below. Ensures shipped rows round-trip & parse faithfully."""
    # cheap pre-guards BEFORE any O(n^2) stdlib-regex work, to bound DoS on pathological rows
    for m in cascade_msgs:
        c = m.get("content") or ""
        if len(c) > 2_000_000:
            return False, "oversized_content"
        if c.count("<tool_call>") > 200:                 # legit max ~99 calls/row
            return False, "too_many_tool_calls"
        if is_tool_result(c) and c.count("<tool_response>") > 1:   # avoid greedy multi-block merge
            return False, "multiple_tool_responses"
    conv = convert_messages(cascade_msgs)
    for m in conv:
        if m["role"] not in VALID_ROLES:
            return False, f"bad_role:{m['role']}"
        if m["role"] != "environment" and "<tool_response>" in (m.get("content") or ""):
            return False, "leftover_tool_response"
    n_src_tr = sum(1 for m in cascade_msgs if is_tool_result(m.get("content") or ""))
    if n_src_tr != sum(1 for m in conv if m["role"] == "environment"):
        return False, "env_count_mismatch"
    src_a = [m for m in cascade_msgs if m["role"] == "assistant"]
    conv_a = [m for m in conv if m["role"] == "assistant"]
    if len(src_a) != len(conv_a):
        return False, "assistant_count_mismatch"
    for sm, cm in zip(src_a, conv_a):
        c = sm.get("content") or ""
        # tag-balance: a value containing </parameter> or </function></tool_call> would truncate silently
        if (c.count("<parameter=") != c.count("</parameter>")
                or c.count("<function=") != c.count("</function>")
                or c.count("<tool_call>") != c.count("</tool_call>")):
            return False, "unbalanced_tool_tags"
        # _CALLP already strips the wrapping newlines; converter emits repr(value) and the parser
        # literal_eval's it back, so orig == parsed iff there was no truncation/corruption.
        orig = [(n, dict(ps)) for n, ps in _calls_from_assistant(c)]
        # balanced-nested protocol tags keep counts balanced but truncate the value silently, and
        # the byte round-trip can't catch it (converter + this gate share _CALLP). Independent guard:
        # reject if any EXTRACTED value still contains a protocol sentinel.
        for _, d in orig:
            for v in d.values():
                if any(s in v for s in _SENTINELS):
                    return False, "sentinel_in_value"
        if not orig:
            if cm.get("function_calls"):
                return False, "spurious_function_calls"
            continue
        parsed = _olmo_parse(cm.get("function_calls"))
        if parsed is None:
            return False, "parser_reject_or_timeout"
        if parsed != orig:
            return False, "roundtrip_mismatch"
    return True, ""


if __name__ == "__main__":
    import argparse, pyarrow as pa, pyarrow.parquet as pq, collections
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quarantine", default=None, help="optional parquet for quarantined rows")
    a = ap.parse_args()
    pf = pq.ParquetFile(a.inp)
    OLMO_MSG = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string()),
                                   ("functions", pa.string()), ("function_calls", pa.string())]))
    other_cols = [c for c in pf.schema_arrow.names if c != "messages"]
    out_schema = pa.schema([(c, pf.schema_arrow.field(c).type) for c in other_cols] + [("messages", OLMO_MSG)])
    w = pq.ParquetWriter(a.out, out_schema, compression="zstd")
    wq = pq.ParquetWriter(a.quarantine, pf.schema_arrow, compression="zstd") if a.quarantine else None
    kept = quar = 0; reasons = collections.Counter()
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg)
        msgs = t.column("messages").to_pylist()
        keep_idx, q_idx, conv_rows = [], [], []
        for i, m in enumerate(msgs):
            ok, why = validate(m)
            if ok:
                keep_idx.append(i)
                c = convert_messages(m)
                conv_rows.append([{"role": x["role"], "content": x.get("content"),
                                   "functions": x.get("functions"), "function_calls": x.get("function_calls")} for x in c])
            else:
                q_idx.append(i); reasons[why] += 1
        if keep_idx:
            cols = {c: t.column(c).take(pa.array(keep_idx)) for c in other_cols}
            cols["messages"] = pa.array(conv_rows, type=OLMO_MSG)
            w.write_table(pa.table(cols, schema=out_schema)); kept += len(keep_idx)
        if wq and q_idx:
            wq.write_table(t.take(pa.array(q_idx))); quar += len(q_idx)
        if rg % 100 == 0:
            print(f"  rg {rg}/{pf.metadata.num_row_groups} | kept {kept:,} quarantined {quar:,}", flush=True)
    w.close()
    if wq: wq.close()
    print(f"DONE: kept {kept:,} | quarantined {quar:,} ({100*quar/max(kept+quar,1):.3f}%) -> {a.out}")
    print("quarantine reasons:", dict(reasons))
