#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

for opti in "--pass incorporate" "" "-O" "--nnef-cycle --nnef-tract-core"
do
    $TRACT_RUN model.onnx $opti run --input-from-bundle io.npz --assert-output-bundle io.npz
done

$TRACT_RUN model.onnx -O dump -q --assert-op-count LstmSeq 4 --assert-op-count Scan 1
