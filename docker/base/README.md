# Base image Dockerfiles (reference)

The deploy images (`docker/Dockerfile.{olmocore,axolotl}`) are **thin layers** on prebuilt base
images. The **olmo-core SFT base** is built from THIS repo's `Dockerfile.olmo-core-sft`, which
**`git clone`s the olmo_core source from the fork (`hav4ik/OLMo-core@olmo3-sft`) at build time** — so
it's reproducible from GitHub with no local checkout (`OLMO_CORE_REF` pins the branch/tag/`main`).
The **official deps base** and the **axolotl base** are still built in their sibling fork repos (heavy,
rarely rebuilt); those Dockerfiles are copied here for documentation and build with their own repo as
context. The forks are the source of truth (`main` = upstream sync, `olmo3-sft` = our divergence:
FA2 sm120 build + the fused-CE z-loss fix).

## Build chain

```
OLMo-core/src/Dockerfile            ──>  olmo-core:cu130-2.10.0      (AI2 official base: torch cu130,
  (= Dockerfile.olmo-core-official)        FA2 sm90;100;120 + FA3 + FA4 + TransformerEngine + liger)
        │
        ▼
docker/base/Dockerfile.olmo-core-sft──>  olmo-core-sft:cu130         (git-clones hav4ik/OLMo-core@olmo3-sft
  (THIS repo, clone-based)                   from GitHub; deps already in the official base)
        │
        ▼
docker/Dockerfile.olmocore          ──>  chankhavu/olmo3-olmocore:cu130   (bootstrap + nvrtc fix)

<axolotl base, see gap below>       ──>  axolotl-olmo3-sft:0.1.0
        │
        ▼
docker/Dockerfile.axolotl           ──>  chankhavu/olmo3-axolotl:cu130

open-instruct/Dockerfile.dataprep   ──>  open-instruct-dataprep:0.1.0     (OFFLINE SFT tokenizer only;
  (= Dockerfile.dataprep)                  --no-deps, no vllm/deepspeed)
```

## Build commands (run from each fork repo)

```bash
# 1. OLMo-core official base (the one edit vs upstream: FA2 +sm120). Makefile tags it by torch/date,
#    then re-tag to the pin the runtime layer expects:
cd OLMo-core && make docker-image            # -> olmo-core:tch2.10.0cu130-<date>
docker tag olmo-core:tch2.10.0cu130-<date> olmo-core:cu130-2.10.0

# 2. OLMo-core SFT base — CLONES the olmo_core source from the fork onto the official base (no local
#    checkout needed; OLMO_CORE_REF pins the branch/tag, default olmo3-sft; context is unused):
cd aimo-olmo3-sft && docker build -f docker/base/Dockerfile.olmo-core-sft \
  --build-arg BASE_IMAGE=olmo-core:cu130-2.10.0 -t olmo-core-sft:cu130 docker/base/

# 3. open-instruct data-prep image (offline tokenizer; version pinned via SETUPTOOLS_SCM_PRETEND_VERSION):
cd open-instruct && docker build -f Dockerfile.dataprep -t open-instruct-dataprep:0.1.0 .

# 4. deploy images (this repo is the context):
cd aimo-olmo3-sft && DOCKERHUB_USER=chankhavu ./docker/build_and_push.sh
```

## ⚠️ Gap: the axolotl base (`axolotl-olmo3-sft:0.1.0`)

There is **no Dockerfile in the workspace** for the axolotl base — it was built ad hoc from the
official `axolotl-uv` (cu130) image + a FlashAttention-3 wheel + NCCL env. To make the axolotl path
fully reproducible, add a `Dockerfile.axolotl-base` here that captures those steps. (The olmo-core
path is the primary/reference engine; axolotl is the fallback.)
