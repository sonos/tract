#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

for rt in "" $TRACT_RUNTIMES
do
    $TRACT_RUN --nnef-tract-core . $rt run --approx approximate \
        --input-from-bundle io.npz --assert-output-bundle io.npz
done

# The deconvolution runs as a gemm on the device, not on the backend's
# deconv kernel nor on the host.
for rt in $TRACT_RUNTIMES
do
    case "$rt" in
        --cuda) asserts="--assert-op-count CudaDeconv 0";;
        --runtime=wgpu) asserts="--assert-op-count WgpuConvTranspose 0 --assert-op-count DeconvSum 0";;
        *) continue;;
    esac
    $TRACT_RUN --nnef-tract-core . $rt dump -q $asserts
done
