#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

$TRACT_RUN --nnef-tract-core . -O run --input-from-bundle io.npz \
    --assert-output-bundle io.npz -q

# A device kernel folds the outer axes onto its grid, so the five have to keep
# their own extents: the pair the operands broadcast on walks different elements
# on each side, and addressing it as one axis would read an operand along an
# axis it does not have.
for rt in $TRACT_RUNTIMES
do
    case "$rt" in -O) continue;; esac
    $TRACT_RUN --nnef-tract-core . $rt run --input-from-bundle io.npz \
        --assert-output-bundle io.npz -q
done
