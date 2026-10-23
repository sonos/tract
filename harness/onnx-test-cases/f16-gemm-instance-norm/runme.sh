#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

$TRACT_RUN model.onnx run --input-from-bundle io.npz --assert-output-bundle io.npz --approx approximate
