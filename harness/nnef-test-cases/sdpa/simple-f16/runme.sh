#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract $CARGO_OPTS --}

rm -rf found
$TRACT_RUN --nnef-tract-core --nnef-tract-transformers . run --input-from-bundle io.npz --steps --assert-output-bundle io.npz --approx approximate
