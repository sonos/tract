#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

$TRACT_RUN --nnef-tract-core . -O run --input-from-npz io.npz --assert-output-bundle io.npz -q

# k reaches the op as an i64 constant, so a GPU takes the op.
for rt in $TRACT_RUNTIMES
do
    case "$rt" in --cuda) backend=Cuda;; --metal) backend=Metal;; --runtime=wgpu) backend=Wgpu;; *) continue;; esac
    $TRACT_RUN --nnef-tract-core . $rt run --input-from-npz io.npz \
        --assert-output-bundle io.npz -q --assert-op-count ${backend}TopK 1
done
