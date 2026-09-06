#!/bin/sh

cd $(dirname $0)
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# Expected values come from onnxruntime. The whole-sequence form associates the
# gate bias sums differently from the decomposed cell, so compare approximately.
# Two batch elements over six timesteps: the step loop has to pick each
# timestep's rows out of the sequence GEMM, not each batch element's.
$TRACT_RUN . -O run --input-from-bundle io.npz --assert-output-bundle io.npz --approx approximate

# The fused op replaces the scan and survives the NNEF round trip.
$TRACT_RUN . -O dump -q --assert-op-count GruSeq 1 --assert-op-count Scan 0

# emit_y travels with the model rather than reverting to its default on reload.
tmp=$(mktemp -d)
sed 's/emit_y = 1/emit_y = 0/' graph.nnef > $tmp/graph.nnef
cp *.dat $tmp/
$TRACT_RUN $tmp -O dump -q --assert-op-count GruSeq 1
$TRACT_RUN $tmp -O run --allow-random-input -q
rm -rf $tmp
