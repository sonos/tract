#!/bin/sh

# ex11-blockified — runs the hand-written blockified reference graph
# against the io.npz produced by ex11-subsampled-section's gen-inputs.
# Both models compute the same thing on the same audio inputs; the
# blockified version just expresses the post-pool SDPA section in
# chunked frame instead of [Tc, Tc].

cd "$(dirname "$0")"
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# Generate the shared io.npz from ex11's source (T=9, s=2).
(cd ../ex11-subsampled-section && python3 gen-inputs.py)
cp ../ex11-subsampled-section/io.npz ./io.npz

# Batch run on s=2 chunks (audio T = 4·s+1 = 9; post-pool Tc = 2·s = 4).
$TRACT_RUN --nnef-tract-core --nnef-tract-transformers --set s=2 . run \
    --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz
