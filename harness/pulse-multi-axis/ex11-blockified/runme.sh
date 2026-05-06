#!/bin/sh

# ex11-blockified — runs the hand-written blockified reference graph
# (the streaming-first form: input [4*s, 4], pool's `+1` boundary
# supplied by an internal after-pad of 1 zero) in batch and pulse.
#
# Both legs compute the same thing on the same audio inputs;
# the post-pool SDPA section is expressed in chunked frame instead
# of [Tc, Tc].  The internal pool pad makes Tc = 2*s clean, which
# divides into [s, P=2] cleanly *and* keeps the streaming-axis dim
# `4*s` linear in s — so pulse-on-s actually goes through.

cd "$(dirname "$0")"
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

python3 gen-inputs.py

# Batch — s=2, audio T = 4*s = 8, post-pool Tc = 2*s = 4.
$TRACT_RUN --nnef-tract-core --nnef-tract-transformers --set s=2 . run \
    --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz

# Pulse — s=1 means each chunk is one transformer chunk = 4 audio frames.
$TRACT_RUN --nnef-tract-core --nnef-tract-transformers . --pulse 's=1' run \
    --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz
