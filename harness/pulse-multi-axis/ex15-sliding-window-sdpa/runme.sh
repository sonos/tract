#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

$TRACT_RUN --nnef-tract-core --nnef-tract-transformers --set T=12 . run \
    --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz

# One chunk per pulse, whatever the pulse: 2 and 3 positions, the latter not
# dividing the window.
for pulse in 2 3
do
    $TRACT_RUN --nnef-tract-core --nnef-tract-transformers . --pulse "T=$pulse" run \
        --approx approximate \
        --input-from-bundle io.npz --assert-output-bundle io.npz
done
