#!/bin/sh

cd $(dirname $0)
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# Check result is as expected
# bug appear only if model optimized and input-fact-from-bundle
# --approx approximate: the f32 output legitimately varies by ~1 ULP with matmul
# kernel selection (reduction order); the default exact-ish Close check is too tight.
$TRACT_RUN --nnef-tract-core ./model.nnef.tgz -O --input-facts-from-bundle ./io.npz run --input-from-bundle io.npz --assert-output-bundle io.npz --approx approximate
