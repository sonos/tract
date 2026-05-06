#!/bin/sh

# ex11-subsampled-section — known-failing reproducer for blockify
# substitute-granularity issue.  Stride-2 max_pool produces post-
# subsample dim `1 + (T-3)/2` which doesn't simplify cleanly under
# blockify's `T → k·S_0` substitution.  Pulse fails inside blockify's
# section rewrite.

cd "$(dirname "$0")"
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

python3 gen-inputs.py

# Batch — passes (verifies the graph compiles + computes correctly).
$TRACT_RUN --nnef-tract-core --nnef-tract-transformers --set T=12 . run \
    --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz

# Pulsified — currently fails.  Audio-frame chunk = 4 (= stride 2 ·
# transformer chunk 2), so pulse value is 4.
$TRACT_RUN --nnef-tract-core --nnef-tract-transformers . --pulse 'T=4' run \
    --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz
