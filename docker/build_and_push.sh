#!/bin/bash
# Build the two deploy images and push them to DockerHub.
# RUN THIS YOURSELF — it needs `docker login` (the agent has no registry credentials by design).
#
# Prereq: the prebuilt BASE images exist locally on the build box:
#   olmo-core-sft:cu130       (AI2 base + olmo_core + FA2 sm90;100;120 + FA3 + FA4)
#   axolotl-olmo3-sft:0.1.0   (official axolotl-uv cu130 + FA3 wheel)
# These are large; build them once on a machine with a GPU/CUDA toolchain, or pull if you've
# already pushed them. The deploy layer here only adds the ~1KB bootstrap, so build is instant.
set -euo pipefail
DOCKERHUB_USER="${DOCKERHUB_USER:-chankhavu}"
TAG="${TAG:-cu130}"
cd "$(dirname "$0")/.."   # repo root = build context (so COPY bootstrap.sh resolves)

echo "==> building ${DOCKERHUB_USER}/olmo3-olmocore:${TAG}"
docker build -f docker/Dockerfile.olmocore --build-arg BASE=olmo-core-sft:cu130 \
    -t "${DOCKERHUB_USER}/olmo3-olmocore:${TAG}" .

echo "==> building ${DOCKERHUB_USER}/olmo3-axolotl:${TAG}"
docker build -f docker/Dockerfile.axolotl --build-arg BASE=axolotl-olmo3-sft:0.1.0 \
    -t "${DOCKERHUB_USER}/olmo3-axolotl:${TAG}" .

echo "==> login + push (you'll be prompted for your DockerHub password / token)"
docker login -u "${DOCKERHUB_USER}"
docker push "${DOCKERHUB_USER}/olmo3-olmocore:${TAG}"
docker push "${DOCKERHUB_USER}/olmo3-axolotl:${TAG}"
echo "==> done. Images: ${DOCKERHUB_USER}/olmo3-{olmocore,axolotl}:${TAG}"
