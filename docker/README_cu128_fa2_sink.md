# OLMo-core trainer: CUDA 12.8 / FA2 / attention-sink / Ulysses-128K

A variant of the olmo-core SFT image built for four requirements:

1. **Attention sink** — a per-head learnable "sink" logit (gpt-oss style) on every layer.
2. **CUDA 12.8** — cu128 devel base + cu128 torch 2.10; nothing needs CUDA 12.9+.
3. **FlashAttention 2** (not FA3) as the runtime attention backend.
4. **128K context** via **Ulysses** context parallelism (all-to-all sequence parallelism).

Image: `chankhavu/olmo3-olmocore:cu128-fa2-sink`.

---

## What changed vs. the production cu130 image

| | cu130 (production) | **cu128-fa2-sink** |
|---|---|---|
| CUDA / torch | 13.0 / 2.10 cu130 | **12.8.1 / 2.10 cu128** |
| FA2 archs | 90;100;120 + 2.8.1 wheel graft for sm_80/86 | **86;90;100;120 from source** (no graft) |
| FA3 | built (Hopper) | built (Hopper), **but runtime defaults to FA2** |
| FA4 (`fa4` extra) | installed via `.[all]` | **dropped** (`OLMO_EXTRAS` = all − fa4) |
| olmo_core source | `olmo3-sft` | **`olmo3-sft-sink`** (adds sinks) |
| default long-context backend | FA2 for ring CP | **FA2 for ring *and* ulysses, and whenever sinks are on** |

The sink mechanism, all-to-all Ulysses CP, and the cu128 base are all upstream-compatible
additions; the cu130 build path is unchanged (the Dockerfile knobs default to the old behavior).

---

## Build (run this yourself — GPU toolchain + `docker login`)

```bash
# 0. PUSH the branches the build/runtime clone from GitHub:
#    - fork:  hav4ik/OLMo-core   branch olmo3-sft-sink   (the sink code; baked into the image)
#    - code:  hav4ik/aimo-olmo3-sft branch olmocore-cu128-fa2-sink (run.sh + sft_scripts; cloned at RUNTIME)
# 1. Build the chain (FA2 source build for 4 archs + FA3 Hopper => long compile):
OLMO_CORE_DIR=../../OLMo-core ./docker/build_cu128_fa2_sink.sh
# 2. Sanity-check, then: docker push chankhavu/olmo3-olmocore:cu128-fa2-sink
```

At **runtime**, point the code clone at this branch until it is merged into the image's default:
`CODE_REF=olmocore-cu128-fa2-sink` (Dockerfile.olmocore's `CODE_BRANCH`).

---

## Enabling sinks + Ulysses at 128K

All via env (read by `olmocore/run.sh` and the `sft_scripts/*`):

```bash
OLMO_USE_SINK=1            # add per-head sinks to every layer (forces the FA2 backend)
OLMO_SINK_INIT=-10.0       # initial per-head sink logit (see "warm start" below); default 0.0
OLMO_CP_STYLE=ulysses      # all-to-all sequence parallelism (vs ring, the default)
SEQ_LEN=131072             # 128K; cp_degree auto = 8 on an 8xH200 node (16384 tok/rank cap)
```

Constraints checked/satisfied for 128K:
- Ulysses needs `cp_degree <= GPUS_PER_NODE` (8 ≤ 8 on one H200 node) and `n_heads % cp_degree == 0`
  (7B: 32 heads, 32B: 40 heads — both divisible by 8).
- Sinks require the **FA2** backend; `run.sh` forces `flash_2` when `OLMO_USE_SINK=1` (FA3/FA4/TE raise).

### Warm start

Stock Olmo-3 was trained without sinks. `OLMO_SINK_INIT=-10.0` makes each sink a near no-op at
step 0 (`o_sink ≈ o`), so training starts ~identical to native Olmo-3 and learns the sink in.
`0.0` = from-scratch. If you warm-start from a **sink-baked** HF checkpoint (measured `s_aux` in
the weights — e.g. from the olmo3_sink fork's `build_init_model.py`), the distcp converter maps
`self_attn.sinks` → `attention.sinks` automatically — just set `OLMO_USE_SINK=1` and the measured
values load (leave `OLMO_SINK_INIT` unset).

### Resuming a partially-trained sink checkpoint (HF → distcp)

For a checkpoint you already trained WITH sinks (HF format, has `self_attn.sinks`), set
`OLMO_USE_SINK=1` and point `HF_MODEL` at it. `run.sh` then passes `--use-sink` to the converter
(`convert_checkpoint_from_hf.py`), which builds the olmo-core model *with* the `sinks` param so the
mapping fills it. Both convert-time and train-time models carry sinks, so the distcp ↔ model params
line up. Requirements/notes:
- The HF `config.json` `model_type` must resolve to the olmo3 arch mapping. If it's `olmo3_sink`
  (trust_remote_code), pass `--model-arch olmo3_32b` (run.sh already does) — the converter keys off
  `--model-arch`, not the HF `model_type`, so the olmo3 sink mapping is used.
- head_dim 128 (÷8) satisfies the sink post-correction requirement.
- To sanity-check the conversion with a logit-parity forward, run the convert **without**
  `--skip-validation` (it uses the eager sink on the torch backend).

---

## How the sink is implemented (FA2)

A sink is an extra key-independent logit `s_h` per head in the softmax denominator. FA2 returns the
no-sink output `o` and `lse = log Σ exp(scores)`; we rescale exactly:

```
o_sink   = o · 1/(1 + exp(s − lse))            # sink contributes 0 to the value mix
lse_sink = lse + softplus(s − lse) = log D     # sink-inclusive lse
```

We overwrite FA2's saved `(out, lse)` in place with `(o_sink, lse_sink)`, so its native backward
returns the **exact** `dq/dk/dv` for sink attention; the closed-form `dsink` is added on top. Under
**Ulysses**, the all-to-all makes each rank's `lse` full-sequence for its head slice, so the
correction is exact per head (the sink is sliced to the rank's contiguous head block). See
`OLMo-core/src/olmo_core/nn/attention/attention_sink.py`.

**Backend support:** **flash_2** (unpacked + packed, non-CP + Ulysses) and **flash_3** (varlen;
non-CP + Ulysses) — same exact post-correction, since FA3's varlen also returns `softmax_lse`. The
**torch** backend has an eager sink (`eager_sink_attention`, the convert-validation / 1b / CPU path).
**Ring CP + sink** raises (ring's lse is incremental across ranks). **FA4/TE** reject sink. **TP +
sink** raises (use FSDP + Ulysses). Dense (non-varlen) FA3 + sink raises (SFT always uses varlen).

---

## Verified vs. to-verify

**Verified locally (RTX 3090, torch 2.11):** `OLMo-core/src/test/nn/attention/attention_sink_test.py`
- Sink math — forward, `dsink`, and the dq/dk/dv identity vs an eager reference (fp32 + bf16).
- **Eager path numerical correctness** (non-circular): `eager_sink_attention` vs the independent
  re-normalization identity `o_nosink · 1/(1+exp(sink−lse))` → fp32 max|Δ| **3.6e-7**; plus the
  no-sink limit (sink→−∞ recovers plain attention) = 0.0.
- HF↔distcp sink conversion round-trip (upload, sink-baked warm-start, stock no-sink — no errors).
- Preset/convert → model wiring: `olmo3_*(use_sink=True, sink_init=…)` materializes
  `blocks.{i}.attention.sinks[n_heads]`; convert `--use-sink` builds the sink model + pre-fills init.
- cu128 torch 2.10 wheels exist; `OLMO_EXTRAS` (all − fa4) matches `pyproject`'s `all`.

**Run on a GPU with flash-attn (H100 for FA3) — the in-kernel test:**
`OLMo-core/src/test/nn/attention/attention_sink_flash_test.py` exercises the REAL FA2/FA3 kernels
with a sink and checks forward + dq/dk/dv/dsink vs the eager reference (skips where flash/Hopper
absent). `python src/test/nn/attention/attention_sink_flash_test.py` or `pytest … -s`.

**Still to verify on the build/GPU box:**
- The above in-kernel test actually passing on FA2 (container/3090) and FA3 (H100).
- Multi-GPU Ulysses cp=8 at 128K with sinks (single-process cp=1 is covered by the head-slice unit test).
- Full image build + `import torch, flash_attn, ring_flash_attn` sanity (see build script tail).
- Converting your partially-trained 32B sink checkpoint (`OLMO_USE_SINK=1`, `HF_MODEL=<it>`).

## Known caveats

- **Ulysses + intra-doc masking + torch.compile** previously tripped a device-side index assert in
  olmo-core (full-sequence `cu_doc_lens` vs a seq-sharded tensor) — validate the 128K packed run
  before a long job, or disable compile / doc-masking to isolate. This is pre-existing olmo-core
  behavior, independent of the sink code.
- `sink_init=0.0` on a from-no-sink warm start is a slow/"dead" start for SFT-time injection; prefer
  a strongly negative init or a sink-baked checkpoint.
