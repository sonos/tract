#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

BATCHIFY='-t batchify(symbol:Some("B"),shared:Some(["scale"]))'

# As exported: one stream, and no batch axis anywhere in the graph.
$TRACT_RUN --nnef-tract-core --set T=5 . run \
    --input-from-bundle io.npz --assert-output-bundle io.npz

# Batchified: the batch axis is on axis 0 of the batched input and of the
# output, and the shared input keeps the shape its caller passes.
$TRACT_RUN --nnef-tract-core . $BATCHIFY dump -q \
    --assert-output-fact "B,T,1,F32"

# Two seats, each with frames of its own: a batchify that folds the seats into
# one another -- the matmul contracting the batch axis away is the way that
# happens -- answers both seats the same and fails here.
$TRACT_RUN --nnef-tract-core . $BATCHIFY --set B=2 --set T=5 run \
    --input-from-bundle io-b2.npz --assert-output-bundle io-b2.npz
