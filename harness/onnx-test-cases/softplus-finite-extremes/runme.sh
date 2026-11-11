#!/bin/sh
set -eu
cd "$(dirname "$0")"
: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}
for dtype in f32 f16
do
    for optimize in '' '-O'
    do
        $TRACT_RUN "$dtype.onnx" $optimize \
            -i "x:7,$dtype=-1000,-10,-1,0,1,10,1000" run --assert-sane-floats \
            --assert-output "y:7,$dtype=0,0.0000454,0.3132617,0.6931472,1.3132617,10.0000454,1000" --approx approximate
    done
done
