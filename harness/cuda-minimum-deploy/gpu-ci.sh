#!/usr/bin/env bash
# Runs tract-cli against a tiny NNEF model inside a Docker container that
# has only the minimum CUDA runtime packages installed — the BOM spelled
# out in cuda/src/context.rs and pinned in the sibling Dockerfile.
#
# Intended to be runnable locally on a GPU workstation so you can catch
# BOM drift without pushing to CI and waiting. The matching CI job is
# `cuda-minimum-deploy` in .github/workflows/crates.yml.
#
# Requirements on the host:
#   - docker + nvidia-container-toolkit
#   - an NVIDIA GPU accessible via --gpus all
#   - a full CUDA toolkit (for building tract-cli — NOT for running)
#
# Usage: harness/cuda-minimum-deploy/gpu-ci.sh

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

HERE="harness/cuda-minimum-deploy"
IMAGE_TAG="tract-cuda-minimal:13-0"
MODEL_DIR="harness/nnef-test-cases/conv-q40/conv_base_kernel1"

echo "==> Building tract-cli (release) on the host"
cargo build --release -p tract-cli

echo "==> Building minimum-BOM runtime image"
docker build -t "$IMAGE_TAG" "$HERE"

echo "==> Running tract inside minimum-BOM container"
docker run --rm --gpus all \
    -v "$PWD/target/release/tract:/usr/local/bin/tract:ro" \
    -v "$PWD/$MODEL_DIR:/model:ro" \
    -w /model \
    -e RUST_LOG="${RUST_LOG:-tract_cuda=info}" \
    "$IMAGE_TAG" \
    tract --nnef-tract-core model.nnef.tgz -O -r cuda run \
        --approx very --input-from-bundle io.npz --assert-output-bundle io.npz

echo "==> OK: tract ran end-to-end inside the minimum-BOM container"
