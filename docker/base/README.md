# Base image Dockerfiles (reference)

The deploy images (`docker/Dockerfile.{olmocore,axolotl}`) are **thin layers** on prebuilt base
images. Those bases are built in the sibling fork repos; the Dockerfiles are copied here so the full
build chain is documented in one place. **They build with their ORIGINAL repo as the context** (they
COPY that repo's source) — not from this repo. Copies here may drift from the forks; the forks are
the source of truth.

## Build chain

```
OLMo-core/src/Dockerfile            ──>  olmo-core:cu130-2.10.0      (AI2 official base: torch cu130,
  (= Dockerfile.olmo-core-official)        FA2 sm90;100;120 + FA3 + FA4 + TransformerEngine + liger)
        │
        ▼
OLMo-core/docker/Dockerfile.runtime ──>  olmo-core-sft:cu130         (+ pinned olmo_core source, baked)
  (= Dockerfile.olmo-core-sft)
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

# 2. OLMo-core SFT base (bakes the olmo_core source onto the official base):
cd OLMo-core && docker build -f docker/Dockerfile.runtime \
  --build-arg BASE_IMAGE=olmo-core:cu130-2.10.0 -t olmo-core-sft:cu130 .

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
