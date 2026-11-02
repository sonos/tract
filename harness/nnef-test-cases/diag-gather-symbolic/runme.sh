#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

# DiagGather survives an NNEF dump with its offset and width still in terms of
# T, and the reloaded graph gives what the original does (--allow-random-input
# is seeded from a fixed constant, so both runs see the same scores).
$TRACT_RUN . --nnef-tract-transformers --set T=5 run --allow-random-input \
    --save-outputs-npz $TMP/reference.npz
$TRACT_RUN . --nnef-tract-transformers dump -q --nnef-dir $TMP/cycled
grep -q 'tract_transformers_diag_gather(scores, offset = (T + -1), out_len = T)' $TMP/cycled/graph.nnef
$TRACT_RUN $TMP/cycled --nnef-tract-transformers --set T=5 run --allow-random-input \
    --assert-output-bundle $TMP/reference.npz
