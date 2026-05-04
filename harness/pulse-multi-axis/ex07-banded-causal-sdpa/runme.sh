#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# Batch
$TRACT_RUN --nnef-tract-core --nnef-tract-transformers --set T=6 . run \
    --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz

# Pulsified — Q·Kᵀ → SMS(banded mask) → attn·V.  At chunk(i) = 0 the
# past j-chunk doesn't exist; today's const all-true block-mask gives
# the wrong softmax denominator there (the streamed pulse output will
# disagree with the batch reference at the boundary).  Fixing this
# requires the chunked mask to faithfully include the in-bounds check
# — see plan in commit message / blockify docs.
$TRACT_RUN --nnef-tract-core --nnef-tract-transformers . --pulse 'T=2' run \
    --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz
