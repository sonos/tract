#!/bin/sh

cd $(dirname $0)
set -ex

: ${TRACT_RUN:=cargo run -p tract $CARGO_OPTS --}

$TRACT_RUN ./model.nnef.tgz --nnef-tract-core run --input-from-bundle ./io.npz --assert-output-bundle ./io.npz -q
