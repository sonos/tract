#!/bin/sh

cd $(dirname $0)
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

$TRACT_RUN --nnef-tract-onnx . -O --input-facts-from-bundle ./io.npz run --input-from-bundle io.npz \
    --assert-output-bundle io.npz --approx exact -q --assert-op-count Resize 0
