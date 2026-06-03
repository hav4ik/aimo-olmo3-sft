#!/usr/bin/env python
"""Probe which FP8 `torch._scaled_mm` configs cuBLASLt actually supports on THIS GPU.

The olmo-core FP8 path (torchao float8) lowers each linear to `torch._scaled_mm`, which
dispatches to cuBLASLt. On sm_120 (RTX PRO 6000 workstation Blackwell) the default
recipe (rowwise + fast_accum) fails with CUBLAS_STATUS_NOT_SUPPORTED — but other scaling
modes may have algorithm coverage. This enumerates the combos so we know which
`OLMO_FP8=<recipe>` (if any) can run here, instead of guessing.

Run inside the container:  python /data/training/code/tools/fp8_probe.py
Maps to OLMO_FP8: scale=tensorwise -> tensorwise ; scale=rowwise -> rowwise / rowwise_with_gw_hp.
"""
import torch

dev = "cuda"
assert torch.cuda.is_available(), "no CUDA device"
cc = torch.cuda.get_device_capability()
print(f"torch {torch.__version__} | {torch.cuda.get_device_name()} | sm_{cc[0]}{cc[1]} | "
      f"cuda {torch.version.cuda}")
print(f"{'fp8':9} {'scale':11} {'fast_accum':10} {'M,N,K':17} result")
print("-" * 78)


def probe(M, N, K, scale, fast_accum, fp8):
    # a: (M,K) row-major fp8 ; b: (K,N) COLUMN-major fp8 (what _scaled_mm wants)
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16).to(fp8)
    b = torch.randn(N, K, device=dev, dtype=torch.bfloat16).to(fp8).t()
    if scale == "tensorwise":
        sa = torch.ones((1, 1), device=dev, dtype=torch.float32)
        sb = torch.ones((1, 1), device=dev, dtype=torch.float32)
    else:  # rowwise: per-row scale on A, per-col scale on B
        sa = torch.ones((M, 1), device=dev, dtype=torch.float32)
        sb = torch.ones((1, N), device=dev, dtype=torch.float32)
    try:
        torch._scaled_mm(a, b, scale_a=sa, scale_b=sb,
                         out_dtype=torch.bfloat16, use_fast_accum=fast_accum)
        torch.cuda.synchronize()
        return "OK"
    except Exception as e:
        return "FAIL: " + str(e).splitlines()[0][:60]


# e4m3 is the forward-GEMM input dtype the recipes use; (16384,4096,4096) is the shape
# from the crash, (4096,)^3 a cheap sanity size.
for fp8 in (torch.float8_e4m3fn, torch.float8_e5m2):
    for scale in ("tensorwise", "rowwise"):
        for fast_accum in (True, False):
            for (M, N, K) in ((4096, 4096, 4096), (16384, 4096, 4096)):
                r = probe(M, N, K, scale, fast_accum, fp8)
                tag = str(fp8).replace("torch.float8_", "")
                print(f"{tag:9} {scale:11} {str(fast_accum):10} {f'{M},{N},{K}':17} {r}")

print("-" * 78)
print("Any OK row => that scaling mode works here. tensorwise OK but rowwise FAIL =>")
print("try training with: OLMO_FORCE_FP8=1 OLMO_FP8=tensorwise PRECISION=fp8")
