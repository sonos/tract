#!/bin/sh

# ex14-reduced-skew: reduced version of ex14-rel-pos-skew-large-table.
#
# Drops the QKV split (uses separate q/k/v inputs) but keeps the full skew
# trick and the same fixed r_pos = variable[2*T_max-1=15, Dh=4].
#
# The pulsification fails at the DynSlice inside the skew trick because:
#   pos_raw = [1, P, 15] (from PulsedConstSlice on r_pos)
#   pos_padded = [1, P, 16]; reshape([1,-1,T=P]) = [1,8,P]
#   dyn_slice(pos_view, begin=1, end=2*P, axis=1, len=2*P-1)
#   → the DynSlice pulsifier sees end > len (condition "end <= len" fails)
#
# Parameters: T=8, P=2, left_chunks=1, W=4, Dh=4, B=1
# r_pos shape = [2*T_max-1, Dh] = [15, 4]

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

# Pulsed run — fails: DynSlice pulsifier "end <= len" condition at pos_sliced.
# TODO: fix the pulsifier to handle the skew trick with a fixed large r_pos.
$TRACT_RUN \
    --nnef-tract-core --nnef-tract-transformers \
    . \
    -t 'pulse(symbol: Some("S"), pulse: "2")' \
    run \
    --input-from-bundle io.npz \
    --assert-output-bundle io.npz \
    --approx very
