#!/bin/sh

cd $(dirname $0)
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

$TRACT_RUN --nnef-tract-onnx . -O run --input-from-bundle io.npz --assert-output-bundle io.npz

$TRACT_RUN --nnef-tract-onnx . dump -q --nnef-graph found
grep -F 'count_symbol = "a"' found
grep -F 'count_symbol = "b"' found
rm found
