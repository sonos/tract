#!/bin/sh

cd $(dirname $0)
set -ex

: ${TRACT_RUN:=cargo run -p tract $CARGO_OPTS --}

# Check result is as expected
# bug appear only if model optimized and input-fact-from-bundle
$TRACT_RUN --nnef-tract-core ./model.nnef.tgz -O --input-facts-from-bundle ./io.npz run --input-from-bundle io.npz --assert-output-bundle io.npz
