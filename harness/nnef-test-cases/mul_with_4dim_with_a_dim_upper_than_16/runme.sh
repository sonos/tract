#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract $CARGO_OPTS --}

$TRACT_RUN . --nnef-tract-core run --assert-output-bundle ./io.npz --input-from-bundle ./io.npz -q
