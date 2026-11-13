#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

$TRACT_RUN . run --input-from-npz io.npz --assert-output-bundle io.npz --approx approximate -q \
    --assert-op-count GeluApproximate 3 --assert-op-count Tanh 0

$TRACT_RUN . -O run --input-from-npz io.npz --assert-output-bundle io.npz --approx approximate -q

for rt in $TRACT_RUNTIMES
do
    case "$rt" in --cuda) backend=Cuda;; --metal) backend=Metal;; --runtime=wgpu) backend=Wgpu;; *) continue;; esac
    $TRACT_RUN . $rt run --input-from-npz io.npz --assert-output-bundle io.npz --approx approximate \
        -q --assert-op-only "${backend}GeluApproximate,DeviceSync*,Const,Source"
done
