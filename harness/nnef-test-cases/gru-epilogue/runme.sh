#!/bin/sh

cd $(dirname $0)
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# Expected values come from onnxruntime. The fused cell associates the gate bias
# sums differently from the decomposed form, so compare approximately.
$TRACT_RUN . -O run --input-from-bundle io.npz --assert-output-bundle io.npz --approx approximate

# The fusion has to survive the NNEF round trip, and nothing decomposed may be
# left inside the scan body.
$TRACT_RUN . -O dump -q --assert-op-count GruEpilogue 1 --assert-op-count Sigmoid 0 --assert-op-count Tanh 0
