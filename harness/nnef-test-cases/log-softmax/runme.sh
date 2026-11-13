#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

$TRACT_RUN --nnef-tract-core . run --input-from-bundle io.npz --assert-output-bundle io.npz

for rt in $TRACT_RUNTIMES
do
    case "$rt" in --cuda|--metal|--runtime=wgpu) ;; *) continue;; esac
    $TRACT_RUN --nnef-tract-core . $rt run --input-from-bundle io.npz \
        --assert-output-bundle io.npz --approx approximate -q \
        --assert-op-only 'CudaLogSoftmax,MetalLogSoftmax,WgpuLogSoftmax,DeviceSync*,Source'
done
