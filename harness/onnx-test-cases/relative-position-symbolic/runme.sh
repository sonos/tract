#!/bin/sh
set -eu
cd "$(dirname "$0")"
: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}
for optimize in '' '-O'
do
    for L in 2 6
    do
        $TRACT_RUN model.onnx $optimize --onnx-ignore-output-shapes --assert 'L>=1' run \
            --input-from-bundle io-$L.npz --assert-output-bundle io-$L.npz
    done
done
