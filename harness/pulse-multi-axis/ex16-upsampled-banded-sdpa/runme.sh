#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

$TRACT_RUN --nnef-tract-core --nnef-tract-transformers --set F=3 . run \
    --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz

# One frame per pulse is 2 mask chunks, three frames are 6.
for pulse in 1 3
do
    $TRACT_RUN --nnef-tract-core --nnef-tract-transformers . --pulse "F=$pulse" run \
        --approx approximate \
        --input-from-bundle io.npz --assert-output-bundle io.npz
done
