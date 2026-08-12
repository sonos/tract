#!/bin/sh

cd $(dirname $0)
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

$TRACT_RUN . -O run --input-from-bundle io.npz --assert-output-bundle io.npz

$TRACT_RUN . -O dump -q --nnef-graph found
grep -F 'concat' found
if grep -q -F 'scatter_nd' found; then
    echo "scatter_nd survived optimisation: the constant-index block rewrite did not fire"
    exit 1
fi
rm found
