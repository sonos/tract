#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# --allow-random-input is seeded from a fixed constant, so `data` is the same
# every run. The graph runs the same STFT twice, once with the frame contiguous
# and once with the batch axis between the time axis and the [re, im] pair, and
# outputs the summed absolute difference: any divergence between the contiguous
# and the strided path makes it non-zero. The reference here is the op's own
# other code path, not a golden value, so the two legs live in one graph rather
# than in an --assert-output-bundle pair.
#
# --assert-op-count pins that both STFTs survive as distinct nodes, so no pass
# folded one leg into the other through the op's axes_mapping and left the
# comparison trivially true.
$TRACT_RUN . run --allow-random-input \
    --assert-op-count STFT 2 \
    --assert-output 'mismatch:1,1,1,1,f32=0'

$TRACT_RUN . -O run --allow-random-input \
    --assert-op-count STFT 2 \
    --assert-output 'mismatch:1,1,1,1,f32=0'
