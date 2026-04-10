#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# Batch mode: concretize S=8 -> pe_table[0:8, :] + input[0:8, :]
$TRACT_RUN --nnef-tract-core . \
    -t 'concretize_symbols(values: {"S": 8})' \
    run --allow-random-input -q

# Streaming compare: pulse=4.  Each step slices pe_table[0:4, :] (constant
# 0.0001) and adds it to the current input chunk.  Because pe_table is uniform,
# the streaming output matches the batch output element-wise.
$TRACT_RUN --nnef-tract-core . --pulse 4 compare \
    --stream --allow-random-input -q
