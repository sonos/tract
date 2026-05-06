#!/bin/sh

# ex12-rel-pos-multi-slope — minimal repro of encoder.p1's blockify failure
# at the relative-position einsum.  Batch passes; pulse fails inside the
# blockify section rewrite at the chunked-Reshape on the rel-pos table
# axis (slope 2k = 4, target k = 2 — slope mismatch, not a constant
# offset, so the affine-trim helper doesn't apply).

cd "$(dirname "$0")"
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

python3 gen-inputs.py

# Batch — passes.
$TRACT_RUN --nnef-tract-core --nnef-tract-transformers --set T=4 . run \
    --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz

# Pulsified — currently fails:
#   "in output_facts invocation for pos_raw.1.blockify_split: Reshape"
#   "1+4*S should be equal to 2*S"
$TRACT_RUN --nnef-tract-core --nnef-tract-transformers . --pulse 'T=2' run \
    --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz
