#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

$TRACT_RUN --nnef-tract-transformers . -t detect_sdpa run --input-from-npz io.npz \
    --assert-output-bundle io.npz --approx approximate -q --assert-op-count SDPA 1 --assert-op-count Softmax 0

$TRACT_RUN --nnef-tract-transformers . -t detect_sdpa -O run --input-from-npz io.npz \
    --assert-output-bundle io.npz --approx approximate -q
