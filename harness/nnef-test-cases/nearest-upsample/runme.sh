#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

$TRACT_RUN --nnef-tract-core . -O run --input-from-npz io.npz \
    --assert-output-bundle io.npz -q --assert-op-count NearestUpsample 1

# The upsample has no device kernel: a GPU runtime expands it back to a
# broadcast rather than leaving a host op in the middle of the graph.
for rt in $TRACT_RUNTIMES
do
    case "$rt" in -O) continue;; esac
    $TRACT_RUN --nnef-tract-core . $rt run --input-from-npz io.npz \
        --assert-output-bundle io.npz -q \
        --assert-op-only 'Cuda*,Metal*,Gpu*,DeviceSync*,Const,Source'
done
