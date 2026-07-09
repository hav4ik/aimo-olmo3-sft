#!/bin/bash
# Build the CUDA-12.8 / FA2 / attention-sink OLMo-core trainer image chain.
#
# RUN THIS YOURSELF on a box with a CUDA toolchain + Docker (the agent has no GPU build box
# or registry creds by design). The FA2 source build (sm_86;90;100;120) + FA3 (Hopper) is a
# long compile.
#
# Everything is CUDA 12.8.1 / cu128 / torch 2.10 (the upstream Dockerfile's own defaults). vs. the
# production cu130 chain the differences are:
#   * cu128 devel base + cu128 torch wheels (nothing needs CUDA 12.9+),
#   * FA2 built for sm_86;90;100;120 from source (no cu13 wheel graft needed -> no fa2-allsm step),
#   * flash-attn-4 DROPPED (OLMO_EXTRAS = all-minus-fa4); FA3 (Hopper) kept but runtime defaults to FA2,
#   * olmo_core source cloned from the `olmo3-sft-sink` branch (adds per-head attention sinks).
#
# Layers (each reuses the previous as BASE):
#   olmo-core:cu128-2.10.0            official deps base (torch/FA2/FA3/TE/ring/liger, cu128, no fa4)
#   olmo-core-sft:cu128              + olmo_core source from the olmo3-sft-sink fork branch
#   chankhavu/olmo3-olmocore:cu128-fa2-sink   + runtime bootstrap + nvrtc/TE ld fix + ring-flash-attn
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# OLMo-core checkout used as the BUILD CONTEXT for the deps base (needs pyproject.toml + src/).
OLMO_CORE_DIR="${OLMO_CORE_DIR:-${REPO_ROOT}/../OLMo-core}"
OLMO_CORE_REF="${OLMO_CORE_REF:-olmo3-sft-sink}"   # fork branch carrying the sink code (must be PUSHED)
DOCKERHUB_USER="${DOCKERHUB_USER:-chankhavu}"

BASE_TAG="olmo-core:cu128-2.10.0"
SFT_TAG="olmo-core-sft:cu128"
DEPLOY_TAG="${DOCKERHUB_USER}/olmo3-olmocore:cu128-fa2-sink"

# Full [all] extras MINUS fa4 (keep in sync with pyproject's `all`).
NO_FA4_EXTRAS="dev,beaker,comet,dion,eval,fla,torchao,transformers,wandb"

echo "==> [1/3] deps base ${BASE_TAG}  (context=${OLMO_CORE_DIR})"
[ -f "${OLMO_CORE_DIR}/pyproject.toml" ] || { echo "!! ${OLMO_CORE_DIR}/pyproject.toml missing; set OLMO_CORE_DIR"; exit 1; }
docker build -f "${REPO_ROOT}/docker/base/Dockerfile.olmo-core-official" \
    --build-arg CUDA_VERSION=12.8.1 \
    --build-arg CUDA_VERSION_PATH=cu128 \
    --build-arg TORCH_VERSION=2.10.0 \
    --build-arg FLASH_ATTN_CUDA_ARCHS="86;90;100;120" \
    --build-arg OLMO_EXTRAS="${NO_FA4_EXTRAS}" \
    -t "${BASE_TAG}" "${OLMO_CORE_DIR}"

echo "==> [2/3] + olmo_core source (${OLMO_CORE_REF}) -> ${SFT_TAG}"
echo "    NOTE: this clones ${OLMO_CORE_REF} from github.com/hav4ik/OLMo-core — PUSH that branch first."
docker build -f "${REPO_ROOT}/docker/base/Dockerfile.olmo-core-sft" \
    --build-arg BASE_IMAGE="${BASE_TAG}" \
    --build-arg OLMO_CORE_REF="${OLMO_CORE_REF}" \
    --no-cache \
    -t "${SFT_TAG}" "${REPO_ROOT}/docker/base/"

echo "==> [3/3] + runtime bootstrap -> ${DEPLOY_TAG}"
docker build -f "${REPO_ROOT}/docker/Dockerfile.olmocore" \
    --build-arg BASE="${SFT_TAG}" \
    -t "${DEPLOY_TAG}" "${REPO_ROOT}"

cat <<EOF

==> done.
    base   : ${BASE_TAG}
    sft    : ${SFT_TAG}
    deploy : ${DEPLOY_TAG}

Sanity-check the image before pushing:
    docker run --rm --gpus all ${DEPLOY_TAG} python /workspace/... # or:
    docker run --rm --gpus all --entrypoint python ${DEPLOY_TAG} - <<'PY'
    import torch, flash_attn
    print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| FA2", flash_attn.__version__)
    assert torch.version.cuda.startswith("12.8"), torch.version.cuda
    import ring_flash_attn  # Ulysses needs only torch all-to-all, but ring stays available
    print("ring_flash_attn OK")
    PY

Push (needs \`docker login\`):
    docker push ${DEPLOY_TAG}
EOF
