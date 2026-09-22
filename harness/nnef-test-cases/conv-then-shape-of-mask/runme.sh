#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# Batch mode: S=8 -> conv output T = 1 + 8/2 = 5 frames; the add is fine.
$TRACT_RUN --nnef-tract-core . \
    -t 'set_symbols(values: {"S": 8})' \
    run --allow-random-input -q

# Streaming compare: pulse=4 -> conv produces 2 frames/step.  The
# tract_core_broadcast (shape=[1,1,S/2+1]) must also produce 2 frames/step.
# The MultiBroadcastTo pulsifier removes the constant boundary term so that
# per-pulse size = substitute(S→P) - substitute(S→0) = (1+P/2) - 1 = P/2.
for rt in "" $TRACT_RUNTIMES
do
    $TRACT_RUN --nnef-tract-core . --pulse 4 $rt compare \
        --stream --allow-random-input -q
done

# Same graph in f16.  The `pulse` stage runs before `-t`, so the f32 pad
# constant is still carried by a PulsePad when the cast lands, and the op has
# to cast it to the datum type it fills.  `run`, not `compare --stream`: the
# reference model stays f32, so its inputs no longer match the pulsed facts.
$TRACT_RUN --nnef-tract-core . --pulse 4 -t f32_to_f16 \
    run --allow-random-input -q
