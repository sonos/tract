#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

$TRACT_RUN --nnef-tract-transformers . -O run --input-from-npz io.npz \
    --assert-output-bundle io.npz -q --assert-op-count ScaledMaskedSoftmax 1

# The device kernels address five axes, and a batched attention softmax reaches
# six of which two carry nothing. A GPU runtime drops those rather than leaving
# a host op in the middle of the graph, which would take the scores off the
# device and back once a layer.
for rt in $TRACT_RUNTIMES
do
    case "$rt" in -O) continue;; esac
    $TRACT_RUN --nnef-tract-transformers . $rt run --input-from-npz io.npz \
        --assert-output-bundle io.npz -q \
        --assert-op-only 'Cuda*,Metal*,Gpu*,DeviceSync*,Const,Source'
done
