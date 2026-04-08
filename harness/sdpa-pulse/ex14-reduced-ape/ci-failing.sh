#!/bin/sh

# ex14-reduced-ape: same constant-RPE failure as ex14, without the skew trick.
#
# r_pos = variable[T_const=8, Dh=4] is a constant locked at T_const=8.
# At batch (S=T=8):  content=[1,8,8] + pos=[1,8,8] → output=[1,8,4] ✓
# At pulse (S=P=2):  content=[1,2,4] (K delayed W=4 from chunk-window ROI)
#                    pos=[1,2,8]     (r_pos stays [8,4] — a constant)
#                    add([1,2,4],[1,2,8]) → broadcast(W=4, T_const=8) → FAILS
#
# The graph IS pulsifiable (output [1,S,4]) once the pulsifier correctly
# extracts the W-wide window from r_pos for the current pulse context.
# TODO: fix the pulsifier to extract the correct W-sized window from r_pos.

cd "$(dirname "$0")"
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

python3 gen-inputs.py

# Batch run — passes at S=T=8.
$TRACT_RUN \
    --nnef-tract-core --nnef-tract-transformers \
    . \
    run \
    --input-from-bundle io.npz \
    --assert-output-bundle io.npz \
    --approx very

# Pulsed run — fails: broadcast(W=4, T_const=8) at the Add node.
# TODO: fix the pulsifier to extract the correct W-sized window from r_pos.
$TRACT_RUN \
    --nnef-tract-core --nnef-tract-transformers \
    . \
    -t 'pulse(symbol: Some("S"), pulse: "2")' \
    run \
    --input-from-bundle io.npz \
    --assert-output-bundle io.npz \
    --approx very
