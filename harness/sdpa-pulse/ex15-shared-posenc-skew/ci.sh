#!/bin/sh

# ex15: encoder-like attention with shared posEnc constant + linear projection + skew trick.
#
# Mimics the Nemotron encoder: posEnc [15, 8] is a constant symmetric RPE table,
# linearPos = posEnc @ W_pos projects it, then the skew trick extracts relative
# position scores.
#
# The key challenge: posEnc is a shared constant (in the real model it has 24
# successors), so PropConst doesn't fold linearPos.  pulsify_qk must evaluate
# the chain at pulsify time via try_compute_const.
#
# Parameters: T=8, P=2, left_chunks=1, W=4, H=2, Dh=4, D_model=8

cd "$(dirname "$0")"
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

python3 gen-inputs.py

# Batch run
$TRACT_RUN \
    --nnef-tract-core --nnef-tract-transformers \
    . \
    run \
    --input-from-bundle io.npz \
    --assert-output-bundle io.npz \
    --approx very

# Pulsed run
$TRACT_RUN \
    --nnef-tract-core --nnef-tract-transformers \
    . \
    -t 'pulse(symbol: Some("S"), pulse: "2")' \
    run \
    --input-from-bundle io.npz \
    --assert-output-bundle io.npz \
    --approx very
