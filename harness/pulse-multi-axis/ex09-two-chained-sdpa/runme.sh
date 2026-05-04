#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# Batch
$TRACT_RUN --nnef-tract-core --nnef-tract-transformers --set T=6 . run \
    --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz

# Pulsified — two chained block-diagonal SDPA blocks.  The recogniser
# finds two quadratic sections (one per block); each is rewritten by
# its own TypedModelPatch.  The shared chunk-id wire is preserved
# (faithful chunkification of the mask chain in both sections).
$TRACT_RUN --nnef-tract-core --nnef-tract-transformers . --pulse 'T=2' run \
    --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz
