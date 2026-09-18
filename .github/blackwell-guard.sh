#!/bin/sh
# Fail unless this agent is really on a Blackwell (SM120) GPU: the kernels this
# box exists to cover are behind __CUDA_ARCH__ guards and a props.major launch
# check, so a job that landed elsewhere would go green having exercised only the
# fallback path.
set -e

nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader

cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
case "$cc" in
    12.*) ;;
    *) echo "compute capability $cc is not Blackwell (want 12.x)" >&2; exit 1 ;;
esac
