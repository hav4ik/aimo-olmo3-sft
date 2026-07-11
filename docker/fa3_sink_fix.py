#!/usr/bin/env python3
"""Patch olmo_core's `_SinkMerge` so the stock-FA3 sink post-correction survives FA3's backward.

The exact per-head sink post-correction (`attention_sink.py::_SinkMerge`) overwrites flash's saved
`(out, lse)` IN PLACE via `.data` so flash's native backward becomes sink-exact. On flash-attn-2 the
backward tolerates this; on flash-attn-3 the stricter autograd version-guard rejects the mutated saved
output ("...modified by an inplace operation ... FlashAttnVarlenFuncBackward ... version 1; expected 0")
and the run crashes at `loss.backward()`.

This wraps the two in-place ops in `torch.autograd._unsafe_preserve_version_counter`, which keeps the
tensor's autograd version unchanged across the mutation — so FA3's saved-tensor check sees version 0 and
accepts it, while the values are still corrected. The math is IDENTICAL (same mul_/add_); this only
changes what autograd's version bookkeeping sees, so it cannot alter gradients (FA2 stays exactly as-is,
and its fwd+bwd-vs-eager unit test must still pass).

Applied at image-build time to the BAKED olmo_core (a thin layer on the deployed image); Python-only, no
recompile. Idempotent + fails loudly if the target moved.
"""
import io
import sys

import olmo_core.nn.attention.attention_sink as m

path = m.__file__
src = io.open(path, encoding="utf-8").read()

if "_unsafe_preserve_version_counter" in src:
    print(f"[fa3fix] already patched: {path}")
    sys.exit(0)

# Wrap each in-place op (match the code prefix only; the trailing comment rides along onto the indented
# line). 8-space indent = inside _SinkMerge.forward.
edits = [
    (
        "        out.data.mul_(inv_scale.unsqueeze(-1).to(out.dtype))",
        "        with torch.autograd._unsafe_preserve_version_counter(out):  # proof-pilot FA3 fix\n"
        "            out.data.mul_(inv_scale.unsqueeze(-1).to(out.dtype))",
    ),
    (
        "        lse_hd.data.add_(log_scale)",
        "        with torch.autograd._unsafe_preserve_version_counter(lse_hd):  # proof-pilot FA3 fix\n"
        "            lse_hd.data.add_(log_scale)",
    ),
]
for old, new in edits:
    if src.count(old) != 1:
        sys.exit(f"[fa3fix] ERROR: expected exactly 1 occurrence of:\n{old}\n(found {src.count(old)}) "
                 f"— attention_sink.py changed; regenerate the patch.")
    src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)

# Compile-check the patched module so a bad edit fails the build, not the training run.
import py_compile  # noqa: E402

py_compile.compile(path, doraise=True)
print(f"[fa3fix] patched + compiled OK: {path}")
