# Minimum runtime image for tract-cuda — validates the BOM documented
# in cuda/src/context.rs. If this image doesn't have what
# `ensure_cuda_runtime_dependencies` + `build_nvrtc_opts` expect, the CI
# job using it will fail, and the BOM has drifted.
#
# Intentionally lean: no nvcc, no cuda-toolkit meta-package, no profiler.
# The nvidia/cuda -runtime- base already ships the runtime libs we need
# (libcudart, libcublas, libnvrtc) held at specific versions. We only
# install the delta: CCCL headers, cudart-dev headers (for cuda_fp16.h
# and math_constants.h), and cuDNN. --allow-change-held-packages lets
# apt bump transitive pins the base image set.
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      --allow-change-held-packages \
      cuda-cccl-13-0 \
      cuda-cudart-dev-13-0 \
      libcudnn9-cuda-13 \
 && rm -rf /var/lib/apt/lists/*
