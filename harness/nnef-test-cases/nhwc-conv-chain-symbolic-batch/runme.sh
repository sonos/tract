#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

$TRACT_RUN . -O run --input-from-npz io.npz --assert-output-bundle io.npz --approx approximate -q

for rt in $TRACT_RUNTIMES
do
    case "$rt" in -O) continue;; --runtime=wgpu) moves=0;; *) moves=2;; esac
    $TRACT_RUN . -t f32_to_f16 $rt run --input-from-npz io16.npz --assert-output-bundle io16.npz \
        --approx super -q --assert-op-count GpuMoveAxis $moves
done
