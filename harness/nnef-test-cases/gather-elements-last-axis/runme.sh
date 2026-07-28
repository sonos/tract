#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# --allow-random-input is seeded from a fixed constant, so `data` is the same
# every run. The graph gathers the same elements twice, once on the contiguous
# last-axis fast path and once forced onto the generic path, and outputs the
# summed absolute difference: any divergence between the two makes it non-zero.
# The two legs live in one graph rather than in an --assert-output-bundle pair
# because the reference here is the op's own other code path, not a golden value.
#
# --assert-op-count pins that both gathers survive as distinct nodes, so no pass
# folded one leg into the other and left the comparison trivially true. It says
# nothing about which internal path each node took; that is what the suite-unit
# gather_elements cases cover.
#
# Both configurations are run because -O is what production uses. GatherElements
# implements only output_facts today, so no pass can touch either leg and the two
# graphs are currently identical; --assert-op-count is what would notice if that
# ever changed.
$TRACT_RUN . run --allow-random-input \
    --assert-op-count GatherElements 2 \
    --assert-output 'mismatch:1,1,1,f32=0'

$TRACT_RUN . -O run --allow-random-input \
    --assert-op-count GatherElements 2 \
    --assert-output 'mismatch:1,1,1,f32=0'
