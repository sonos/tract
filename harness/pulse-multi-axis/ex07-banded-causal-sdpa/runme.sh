#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# Batch
$TRACT_RUN --nnef-tract-core --nnef-tract-transformers --set T=6 . run \
    --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz

# Pulsified — Q·Kᵀ → SMS(banded mask) → attn·V.  At chunk(i) = 0 the
# past j-chunk doesn't exist.  Blockify chunkifies the mask construction
# faithfully (Sub/Ge/Le/And replayed in chunked form, with WindowOnAxis
# on the contracted side using a sentinel pad value so the band predicate
# evaluates "out of band" on boundary slots), so SMS sees a chunked mask
# that agrees with the batch reference at every chunk including chunk 0.
$TRACT_RUN --nnef-tract-core --nnef-tract-transformers . --pulse 'T=2' run \
    --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz
