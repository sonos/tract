#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

$TRACT_RUN --nnef-tract-core --nnef-tract-transformers . run \
    --input-from-bundle io.npz \
    --assert-output-bundle io.npz \
    --approx approximate

for rt in $TRACT_RUNTIMES
do
    case "$rt" in --cuda) backend=Cuda;; *) continue;; esac
    $TRACT_RUN --nnef-tract-core --nnef-tract-transformers . $rt run \
        --input-from-bundle io.npz \
        --assert-output-bundle io.npz \
        --approx approximate \
        --assert-op-count ${backend}Mamba2Recurrent 1
done
