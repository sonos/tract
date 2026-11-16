#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

$TRACT_RUN . -t block_quant run --input-from-npz io.npz --assert-output-bundle io.npz --approx approximate -q

for rt in $TRACT_RUNTIMES
do
    case "$rt" in --runtime=wgpu) ;; *) continue;; esac
    $TRACT_RUN . -t block_quant $rt run --input-from-npz io.npz --assert-output-bundle io.npz \
        --approx approximate -q --assert-op-count WgpuGemvPair 0
    $TRACT_RUN . $rt run --input-from-npz io.npz -q --assert-op-count WgpuGemvPair 1
done
