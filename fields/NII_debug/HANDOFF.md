# HANDOFF — OLMo-3 SFT container, v2.1 (NII attempt 2)

Self-contained status + how-to-continue for a fresh agent. Last updated 2026-06-07.

## 1. What this project is
SFT training **container images** to fine-tune **Olmo-3 7B** (32B next) on math-proof CoT data with
**olmo-core**, for a **Fields Model submission to NII/ABCI** (Japan, Singularity/Apptainer cluster).
Two delivery targets from one image: a **`.sif`** (Singularity, for NII — bakes credentials) and a
**docker image** (for VastAI/H200 testing — clean, creds via `-e`).

## 2. Current state (2026-06-07)
- **v2.1 shipped to NII** and **confirmed running on 4×H200 (VastAI)** — clears the compile and trains.
- v2.1 fixes the run_01 crash (`PermissionError: nvcc`) + a follow-on cold-cache crash
  (`FileNotFoundError: '9.0'`). Both root-caused, fixed, and verified old-breaks/new-works on docker
  **and** `.sif` with cold caches. See §4.

## 3. Artifacts & rollback points (FROZEN snapshot = "nii-attempt2")
| What | Where |
|---|---|
| Docker image (current) | `chankhavu/olmo3-sft-v2.1:cu130-allsm` — digest `sha256:8b455a26…` |
| Docker image (frozen snapshot) | `chankhavu/olmo3-sft-v2.1:nii-attempt2` (identical, immutable tag) |
| `.sif` (what NII has) | `s3://aimo-proof-pilot-jp-786106244389-ap-northeast-1-an/containers/olmo-sft-v2.1-allsm.sif` (Tokyo, ap-northeast-1) |
| Local `.sif` | `./olmo-sft-v2.1-allsm.sif` (4.8 GB, secrets baked — gitignored) |
| Git snapshot | branch `nii-attempt2` @ commit `94a9b6a` (GitHub `hav4ik/aimo-olmo3-sft`) |
| Working branch | `olmo-sft-32b` (continue here; roll back to `nii-attempt2` if needed) |
| **FROZEN production** | branch `olmo3-sft` — **do NOT touch** |

## 4. What v2.1 changed (and why) — details in the docs below
The image originally shipped **no nvcc**; on NII a HOST nvcc on a `noexec` mount got picked up via
`PATH` and `torch.compile` crashed every rank (PyTorch catches a *missing* nvcc but not a
*non-executable* one). The fix + hardening:
- **Ship our own nvcc** (CUDA 13.0.88 pip wheels) at `/opt/fields/bin/nvcc`, first on `PATH`.
- **(A)** baked `PATH` reorder — populated in-image dirs before empty host-fillable ones.
- **(B)** `fields/train.py` re-pins in-image dirs first **at runtime** (survives `SINGULARITYENV_PATH` /
  `--env PATH=` overrides). PATH-only — does NOT touch `MASTER_ADDR`/`NCCL_*`/etc.
- **(C)** `ENV CC=/usr/bin/gcc CXX=/usr/bin/g++ NVCC_PREPEND_FLAGS='-ccbin /usr/bin/gcc'` — pin the
  compilers so a host bind can't shadow them.
- **CC footgun fix:** `olmocore/run.sh` + `axolotl/run.sh` stored the GPU compute-cap in a var named
  **`CC`** (reserved C-compiler env var). Once `ENV CC` exported it, `CC="9.0"` leaked into Triton's
  `$CC` compiler lookup → `FileNotFoundError: '9.0'` on **cold-cache** compiles (deterministic on a
  fresh H200). Renamed the var to **`GPU_CC`**. (Triton reads `$CC` as the compiler — never reuse it.)
- **`.dockerignore`:** hard-guards `SECRETS.json` out of the PUBLIC docker build (the `.sif` build is
  Singularity, unaffected, and still bakes it via `%files`).

**Training/checkpointing behavior is UNCHANGED** — same compiler binaries (`/usr/bin/gcc-11`/`g++-11`),
same FP8/attn arch gating, SFT scripts byte-identical. nvcc is inert (only a `--version` probe).

**Read these for the full story (all in `fields/NII_debug/`):**
- `NII_REPORT.md` — the email-ready root cause + reproduction proof (sent to NII).
- `REPRODUCTION.md` + `repro_full_run.log`/`run_01.log` — how the crash was reproduced.
- `nvcc_hypotheses.md` — the diagnosis hypotheses + the host diagnostic.
- `v2.1_email_note.md` — the "what we guard against" note sent to NII (nvcc + gcc/ptxas/libs).
- `compare_OLD.log`/`compare_NEW.log`/`compare_OLDsif.log`/`ccfix_test.log` — old-breaks/new-works evidence.

## 5. Build & deploy (exact process)
Build chain: `olmo-core-sft:cu130-allsm` → `chankhavu/olmo3-olmocore:cu130-allsm` (base) → fields layer.
```bash
# 1) Build the docker image (the build script moves SECRETS.json aside; .dockerignore is a 2nd guard)
docker build -f fields/Dockerfile --build-arg BASE=chankhavu/olmo3-olmocore:cu130-allsm \
    -t chankhavu/olmo3-sft-v2.1:cu130-allsm .
#    (verify clean: docker run --rm --entrypoint sh <img> -c 'test -f /app/SECRETS.json && echo BAD || echo clean')
docker push chankhavu/olmo3-sft-v2.1:cu130-allsm

# 2) Build the .sif (bakes fields/SECRETS.json via the .def %files). Has a mksquashfs-bug fallback.
bash fields/build_sif.sh olmo-sft-v2.1-allsm.sif chankhavu/olmo3-sft-v2.1:cu130-allsm fields/olmo-sft-v2.1-allsm.def

# 3) Upload the .sif to S3 (NII delivery)
aws s3 cp olmo-sft-v2.1-allsm.sif \
    s3://aimo-proof-pilot-jp-786106244389-ap-northeast-1-an/containers/olmo-sft-v2.1-allsm.sif
```
**SECRETS handling:** `fields/SECRETS.json` (real HF+W&B creds) is gitignored AND dockerignored; it is
baked ONLY into the local `.sif`. Never push a docker image built with it to a public registry, and
never commit it or any `*presigned*` file. Verify the docker image is clean before any push.

## 6. How to verify (test harness in `fields/NII_debug/`)
- `nvcc_selfcheck.sh diagnose <sif|img>` / `... test <sif|img>` — fast: plants a non-exec nvcc, runs a
  cold `torch.compile`, PASS/FAIL. (PASS on v2.1, FAIL on the old v2.)
- `run_full_container_test.sh` — full entrypoint run (1B, cold cache, non-exec nvcc bound). Knobs:
  `SIF=`, `MAX_STEPS=`, `RUN_SUFFIX=`, `INJECT_BROKEN_NVCC=0/1`.
- **Cold cache matters:** the CC/nvcc bugs only surface on a COLD Triton/inductor cache. Clear it:
  `rm -rf /mnt/data/sif-test/tmp/olmo-sft/work/node/*/{triton,inductor}` and/or set
  `--env TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`. Local test scratch: `/mnt/data/sif-test/tmp` (bound to /tmp).
- Local GPUs: 2×RTX-3090 (sm_86). Production is 8×H200 (sm_90) — local can't fully exercise sm_90 paths.

## 7. Open items / next steps
- **NII delivery polish (not done):** generate a v2.1 **presigned URL** to email; refresh the bucket's
  `README.md` + upload the v2.1 `.def` (bucket still has v2 docs); **remove the old broken
  `containers/olmo-sft-v2-allsm.sif`** so a stale URL can't grab it.
- **Watch-item — distcp checkpoint-save flake:** an intermittent `tmpXXX.distcp` rename
  `FileNotFoundError` in olmo-core's `filesystem.py` (UNTOUCHED code). Didn't reproduce on clean runs;
  not from our changes. If it recurs on the real H200 run, investigate.
- **Confirm on H200:** the first checkpoint SAVE + HF upload on H200 hadn't been confirmed at handoff
  time (training was confirmed). Watch those milestones.
- **32B scale-up:** see `SCALEUP_32B.md` + the `olmo_32b_{bf16,fp8}` recipes (lr 5e-5, 1M batch). The
  32B `.sif` is ~64 GB — revisit checkpoint-upload timeout/poll before sending NII a 32B run.

## 8. Conventions to respect (IMPORTANT)
- **Branches:** `olmo3-sft` = FROZEN production, do NOT touch. Work on `olmo-sft-32b`.
- **Ask before any `git commit` / `docker push`** — don't do them eagerly.
- **Training stability/safety is the top priority.** Do NOT modify the frozen 7B SFT script
  (`olmocore/sft_scripts/Olmo-3-7B-SFT-local.py`); the 32B is a verbatim clone + arch swap.
- H200 launch shapes (per memory): bf16 → `--max-tokens-per-rank 16384 --olmo-ac-budget 1.0` (cp4);
  fp8 → `32768 / 0.8` (cp2). The README production recipe assumes the full **8×H200** node.
- The agent's auto-memory (`MEMORY.md` + linked notes) has the durable facts; this file is the detailed
  handoff. The key memory note is `nvcc-host-dep-fix-v21`.
