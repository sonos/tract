#!/bin/sh

cd $(dirname $0)
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# A projected GruSeq takes the input-side product as x and has no w input. It
# loads, stays one fused op once optimized (codegen packs R from its projected
# slot), runs, and keeps its form through an NNEF round trip.
$TRACT_RUN . -O dump -q --assert-op-count GruSeq 1
$TRACT_RUN . -O run --allow-random-input -q
$TRACT_RUN . dump -q --nnef-dir roundtrip
grep -q "projected = 1" roundtrip/graph.nnef
$TRACT_RUN roundtrip -O run --allow-random-input -q
rm -rf roundtrip
