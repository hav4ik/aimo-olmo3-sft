# Running Olmo-3 SFT (cu128 / FA2 + in-kernel FA3 sink container)

Image: `chankhavu/olmo3-olmocore:cu128-fa2-sink` — CUDA 12.8, torch 2.10, Ulysses CP, FP8 removed.

**Attention sink is applied via exact post-correction on both FA2 and FA3** (forward + dq/dk/dv/dsink
match the eager reference bit-for-bit up to fp rounding). In-kernel FA3 is **not** active on this
image: torch 2.10 compiles flash-attn 3's stable-ABI `flash_api_stable.cpp`, but the in-kernel sink
patch targets `flash_api.cpp` (used only by torch < 2.9.0.dev, which is what Yi-Chia's fork trains on).
Post-correction is numerically identical for a fresh SFT; enabling true in-kernel FA3 would require
porting the sink through the stable ABI. See `has_fa3_sink_kernel()` — it correctly reports `False` here.

Everything below is **on the GPU node** (needs `docker` + NVIDIA runtime). You drive it with the
host-side launcher `olmocore/launch.sh` — you never write a long `docker run` by hand.

---

## One-time setup on the node

```bash
# 1. get the launcher (this repo, the cu128 branch)
git clone -b olmocore-cu128-fa2-sink https://github.com/hav4ik/aimo-olmo3-sft
cd aimo-olmo3-sft

# 2. get the image
docker pull chankhavu/olmo3-olmocore:cu128-fa2-sink

# 3. secrets (private model + dataset)
export HF_TOKEN=hf_...          # required
export WANDB_API_KEY=...        # optional (loss curves)
```

---

## Run training — ONE command

The container converts the HF model → distcp on the first run (once per data dir), then trains —
no separate convert step.

**128-token smoke** (validate the whole path on 8× H100; ~fits, ~fast):
```bash
./olmocore/launch.sh --data /DATA/olmo-run --seq-len 128 --max-steps 10 --gbs 1024 --max-tokens-per-rank 128
```

**Real run** (drop the smoke overrides; set the real recipe). Use the default **ring** CP —
`--cp-style ulysses` is known-broken for this config (device-side index assert: Ulysses passes
full-sequence `cu_doc_lens` into a per-rank-sharded tensor). See docs/attention-sink-lifecycle.md.
```bash
./olmocore/launch.sh --data /DATA/olmo-run --seq-len 65536 --epochs 2 --max-tokens-per-rank 16384
```

Defaults are already this model (`chankhavu/yccchen-olmo3-deploy`, its deepseek tokenizer + YaRN, and
`OLMO_USE_SINK=1`). Override anything with a flag — `./olmocore/launch.sh --help` lists them all;
add `--dry-run` to print the `docker run` without running it.

> **Note on 256K / 32B:** it does **not** fit on 8× H100 (the 32B model + optimizer alone is ~64 GB/GPU
> sharded, leaving no room for 256K activations). Use ≤64K on 8× H100, or go multi-node for 256K.

---

## Run the unit tests — ONE command

Runs the attention-sink correctness suite **inside the container on this GPU** — most importantly the
**FA3 in-kernel sink** vs the eager reference, and **FA2 post-correction vs FA3 in-kernel** (the two
production paths must agree on forward + dq/dk/dv/dsink):

```bash
./olmocore/launch.sh --test
```

What it runs (each self-skips backends not present, so it validates what the image + GPU actually have):
- `attention_sink_test.py` — sink math vs an eager reference (fp32 + bf16), CPU-only.
- `attention_sink_flash_test.py` — **real FA2/FA3 kernels + sink vs eager, and FA2-correction vs FA3-in-kernel** (needs a GPU; FA3 needs Hopper).
- `attention_sink_ulysses_test.py` — Ulysses CP + sink parity (multi-rank, gloo/CPU).

Expected tail: `ALL ATTENTION-SINK TESTS PASSED`. If the FA2-vs-FA3 check fails, the two sink paths
diverged — send me the `max|Δ|` numbers it prints.

You can also run this **automatically before every training run** with `-e OLMO_ATTN_SELFCHECK=1`
(rank-0 pre-flight; aborts the run if the kernels disagree) — e.g.
`./olmocore/launch.sh --data /DATA/olmo-run … --env OLMO_ATTN_SELFCHECK=1`.

---

## Multi-node (for the real 256K/32B run later)

`launch.sh` takes `--nnodes N --node-rank R --master-addr HOST --master-port P`; your cluster's
launcher sets those per node (rank 0 = the rendezvous host). Everything else (image, flags) is the
same on every node.
