#!/usr/bin/env bash
# Build the Fields submission .sif from the LOCAL docker image, baking fields/SECRETS.json.
#
# Why this isn't a one-liner: apptainer/SingularityCE 1.5.1 bundles mksquashfs 4.7.5 (2026-03),
# which crashes on our ~12GB rootfs ("Bug in orderer" / SIGSEGV in the fragment orderer). So we
# try the normal build first, and if it fails we fall back to: build a sandbox (no squashfs) ->
# compress it with the STABLE system mksquashfs (e.g. 4.5) -> assemble the SIF by hand. The
# runscript/labels/env live inside the rootfs (/.singularity.d), so a squashfs-only SIF still runs.
#
# Usage (run from REPO ROOT so the %files path resolves):
#   bash fields/build_sif.sh [OUTPUT.sif] [DOCKER_TAG]
# Defaults: OUTPUT=olmo3-fields_cu130.sif  DOCKER_TAG=chankhavu/olmo3-fields:cu130
set -euo pipefail

OUT="${1:-olmo3-fields_cu130.sif}"
TAG="${2:-chankhavu/olmo3-fields:cu130}"
DEF="${3:-fields/olmo3-fields.def}"   # pass a specific .def as $3 (its `From:` selects the docker image)
SB="$(mktemp -d /tmp/fields_sb.XXXX)"
SQUASH="$(mktemp -u /tmp/rootfs.XXXX.squashfs)"

cleanup() { rm -rf "$SB" "$SQUASH"; }
trap cleanup EXIT

[ -f "$DEF" ] || { echo "ERROR: run from repo root ($DEF not found)"; exit 1; }
[ -f fields/SECRETS.json ] || echo "WARN: fields/SECRETS.json absent -> SIF will have no baked creds (env/bind at runtime)"

echo ">> Attempt 1: direct singularity build (works if apptainer's mksquashfs is healthy)"
rm -f "$OUT"
if singularity build --fakeroot "$OUT" "$DEF" 2>/tmp/build1.log; then
  echo ">> Direct build succeeded."; singularity sif list "$OUT"; exit 0
fi
echo ">> Direct build failed (likely bundled-mksquashfs bug); falling back to manual route."
grep -iE "orderer|squashfs|139|FATAL" /tmp/build1.log | tail -3 || true

echo ">> Step 1/3: build sandbox (no squashfs)"
rm -rf "$SB"
singularity build --fakeroot --sandbox "$SB" "$DEF"

echo ">> Step 2/3: compress with system mksquashfs ($(/usr/bin/mksquashfs -version 2>/dev/null | head -1))"
/usr/bin/mksquashfs "$SB" "$SQUASH" -noappend -all-root -comp gzip

echo ">> Step 3/3: assemble SIF (primary system partition, squashfs, amd64)"
rm -f "$OUT"
singularity sif new "$OUT"
singularity sif add "$OUT" "$SQUASH" --datatype 4 --parttype 2 --partfs 1 --partarch 2

echo ">> Done."; singularity sif list "$OUT"; ls -lh "$OUT"
echo ">> Verify: singularity run $OUT --help   (and: singularity exec --nv --containall --bind <host>:/tmp $OUT python /app/smoke_test.py)"
