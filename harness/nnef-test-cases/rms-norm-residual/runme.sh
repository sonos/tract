#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

$TRACT_RUN --nnef-tract-core . -O run --input-from-npz io.npz --assert-output-bundle io.npz \
    --approx approximate -q

# Every GPU runtime folds the residual add into its RmsNorm kernel, which hands the sum back as a
# second output for the residual stream: the only add left to dispatch is the next block's.
for rt in $TRACT_RUNTIMES
do
    case "$rt" in --cuda) backend=Cuda;; --metal) backend=Metal;; --runtime=wgpu) backend=Wgpu;; *) continue;; esac
    $TRACT_RUN --nnef-tract-core . $rt run --input-from-npz io.npz --assert-output-bundle io.npz \
        --approx approximate -q --assert-op-count ${backend}RmsNorm 1 \
        --assert-op-count ${backend}Add 1
    $TRACT_RUN --nnef-tract-core . -t f32_to_f16 $rt run --input-from-npz io16.npz \
        --assert-output-bundle io16.npz --approx super -q --assert-op-count ${backend}RmsNorm 1 \
        --assert-op-count ${backend}Add 1
done
