#!/usr/bin/env python
"""Microbenchmark FP8 (tensorwise e4m3 `_scaled_mm`) vs BF16 matmul on THIS GPU.

Shows the raw tensor-core speedup of the FFN GEMMs *in isolation* — i.e. the FP8 win
BEFORE it gets masked by PCIe context-parallel comm in an end-to-end training step.
This is the number to look at when asking "is FP8 worth it on this hardware?"; the
end-to-end step time on a cp=4 PCIe box is comm-bound and will NOT reflect this.

Run inside the container:  python /data/training/code/tools/fp8_bench.py
Note: GEMM-only (no bf16->fp8 cast). The cast/scale is extra per-step overhead in real
training, so end-to-end FP8 speedup <= the ratios below even when compute-bound.
"""
import torch

dev = "cuda"
assert torch.cuda.is_available()
cc = torch.cuda.get_device_capability()
print(f"torch {torch.__version__} | {torch.cuda.get_device_name()} | sm_{cc[0]}{cc[1]} | cuda {torch.version.cuda}")
E4 = torch.float8_e4m3fn


def bench(fn, iters=50, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms/iter


def make(M, N, K):
    a16 = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    b16 = torch.randn(N, K, device=dev, dtype=torch.bfloat16)  # b16.t() -> (K,N) col-major
    a8, b8 = a16.to(E4), b16.to(E4)
    sa = torch.ones((1, 1), device=dev, dtype=torch.float32)
    sb = torch.ones((1, 1), device=dev, dtype=torch.float32)
    bf16 = lambda: torch.mm(a16, b16.t())
    fp8 = lambda: torch._scaled_mm(a8, b8.t(), scale_a=sa, scale_b=sb,
                                   out_dtype=torch.bfloat16, use_fast_accum=True)
    return bf16, fp8


# Olmo-3-7B is hidden=4096; FFN GEMMs at 16384 tok/device are (16384,4096)x(4096,I) and back.
# I (intermediate) for the 7B is ~11008-ish; a few shapes bracket it + a big square for peak.
SHAPES = [(16384, 4096, 4096), (16384, 11008, 4096), (16384, 4096, 11008), (8192, 8192, 8192)]
print(f"{'M,N,K':20} {'bf16 ms':>9} {'fp8 ms':>9} {'speedup':>8} {'bf16 TF/s':>10} {'fp8 TF/s':>10}")
print("-" * 72)
for (M, N, K) in SHAPES:
    bf16, fp8 = make(M, N, K)
    tb, tf = bench(bf16), bench(fp8)
    flop = 2 * M * N * K
    print(f"{f'{M},{N},{K}':20} {tb:9.3f} {tf:9.3f} {tb / tf:7.2f}x {flop / tb / 1e9:10.1f} {flop / tf / 1e9:10.1f}")
print("-" * 72)
print("speedup = bf16_ms / fp8_ms for the GEMM alone. End-to-end FP8 gain is this, MINUS")
print("cast/scale overhead, TIMES the fraction of step time that is FFN compute (small at cp=4/PCIe).")
