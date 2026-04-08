#!/bin/sh

# ex14: Transformer-XL relative-position attention with skew trick, left_chunks=1.
# The position table r_pos is a FIXED VARIABLE of shape [2*T-1, Dh] = [15, 4].
# This models the Nemotron encoder where the RPE table is pre-computed for the
# full sequence length T and loaded as a constant (rather than being dynamically
# sliced from a larger table as in ex12/ex13).
#
# Parameters: B=1, T=8, P=2, left_chunks=1, W=(left_chunks+1)*P=4, Dh=4, H=1
#
# At batch time (S=T=8):
#   pos_raw = Q @ r_pos^T = [1, 8, 15]; skew → pos_scores = [1, 8, 8] ✓
#
# At pulse time (S=P=2):
#   r_pos is still [15, 4] — it's a constant; the DynSlice was folded away
#   before pulsification, so the DynSlice pulsifier never fires.
#   In ex12/ex13, R = DynSlice(r_full, center-S, len=2*S-1) shrinks to
#   [2*P-1=3, 4] at pulse time and the DynSlice pulsifier (Case B) adjusts
#   begin/end for the window.  Here there is no DynSlice to adjust.
#
#   pos_raw = Q[1,P,D] @ r_pos[15,D]^T = [1, P=2, 15]
#   After skew + existing slice-extension fix: pos_scores shape = [1,P,W=4] ✓
#   content_scores shape = [1,P,W=4] ✓
#   Shapes match, so no pulsification error — but VALUES are wrong.
#   The slice-extension fix picks the wrong rows from pos_view because r_pos
#   was not re-centered for the windowed pulse context.
#   Result: pulsed run produces wrong output (≈97% of values are outliers).
#
# compare --stream reveals a deeper problem: the reference model itself fails
# at the stream_dim the harness uses (stream_dim = delay + 3*P + P/2 = 10),
# because pos_view = reshape([1,S,16],[1,-1,S]) = [1,16,S] has only 16 rows
# but the skew slice tries end = 2*S = 20.  The reference batch model is only
# valid for S ≤ 8 (the T it was built for).  This mirrors the Nemotron
# encoder: the RPE table is locked to the full-sequence T, so the batch graph
# cannot be evaluated at a longer stream than originally intended.
#
# The fix requires the pulsifier to recognise that r_pos is a constant RPE
# table centred at position T-1, and to re-extract the W-sized window of rows
# appropriate for the current pulse's context, adjusting for the lookback.

cd "$(dirname "$0")"
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

python3 gen-inputs.py

# Batch (reference) run — passes at S=T=8.
$TRACT_RUN \
    --nnef-tract-core --nnef-tract-transformers \
    . \
    run \
    --input-from-bundle io.npz \
    --assert-output-bundle io.npz \
    --approx very

# Pulsed run — currently produces wrong values (≈97% outliers) because r_pos
# is a constant [15,4] and cannot be adjusted for the windowed context.
# TODO: fix the pulsifier to re-extract the correct window from r_pos.
$TRACT_RUN \
    --nnef-tract-core --nnef-tract-transformers \
    . \
    -t 'pulse(symbol: Some("S"), pulse: "2")' \
    run \
    --input-from-bundle io.npz \
    --assert-output-bundle io.npz \
    --approx very
