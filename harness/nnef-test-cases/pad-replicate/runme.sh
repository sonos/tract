#!/bin/sh

cd $(dirname $0)
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

$TRACT_RUN . run --input-from-bundle io.npz --assert-output-bundle io.npz
$TRACT_RUN . dump --nnef-graph found.nnef
grep -q "border = \"replicate\"" found.nnef
rm found.nnef
