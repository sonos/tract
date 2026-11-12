#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

$TRACT_RUN --nnef-tract-core --nnef-tract-transformers . run \
    --input-from-bundle io.npz --assert-output-bundle io.npz --approx approximate

# The GPU kernels take no bias: the op stays on the CPU, and stays correct.
for rt in $TRACT_RUNTIMES
do
    case "$rt" in --cuda|--metal|--runtime=wgpu) ;; *) continue;; esac
    $TRACT_RUN --nnef-tract-core --nnef-tract-transformers . $rt run \
        --input-from-bundle io.npz --assert-output-bundle io.npz --approx approximate \
        --assert-op-count CausalConv1dUpdate 1
done
