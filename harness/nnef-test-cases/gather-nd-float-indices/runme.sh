#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

for case in rounded truncated
do
    $TRACT_RUN --nnef-tract-core $case -O run --input-from-npz $case/io.npz \
        --assert-output-bundle $case/io.npz -q
done

# A GPU holds no TDim, nor on wgpu 64-bit integers: rounded coordinates are recomputed in f32 so
# the gather and everything feeding it run on the device. Truncated ones are not exact that way,
# and their gather stays on the host.
for rt in $TRACT_RUNTIMES
do
    case "$rt" in --cuda) backend=Cuda;; --metal) backend=Metal;; --runtime=wgpu) backend=Wgpu;; *) continue;; esac
    $TRACT_RUN --nnef-tract-core rounded $rt run --input-from-npz rounded/io.npz \
        --assert-output-bundle rounded/io.npz -q --assert-op-count ${backend}GatherNd 1 \
        --assert-op-only "${backend}*,Gpu*,DeviceSync*,Const,Source"
    $TRACT_RUN --nnef-tract-core truncated $rt run --input-from-npz truncated/io.npz \
        --assert-output-bundle truncated/io.npz -q --assert-op-count GatherNd 1
done
