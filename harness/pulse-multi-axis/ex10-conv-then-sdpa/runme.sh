#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# Batch
$TRACT_RUN --nnef-tract-core --nnef-tract-transformers --set T=6 . run \
    --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz

# Pulsified — max_pool(kernel=3, padding=1) on the queries, then a
# block-diagonal SDPA section.  At pulse time the pool emits a
# streaming-axis Delay of 1; the downstream SDPA section is rewritten
# in chunked form by Blockify and inherits that delay through the
# normal pulse machinery.
$TRACT_RUN --nnef-tract-core --nnef-tract-transformers . --pulse 'T=2' run \
    --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz
