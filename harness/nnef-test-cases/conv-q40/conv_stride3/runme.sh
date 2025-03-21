#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract $CARGO_OPTS --}

$TRACT_RUN --nnef-tract-core model.nnef.tgz -O run --approx very --input-from-bundle io.npz --assert-output-bundle io.npz
