#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

tmp=`mktemp -d`
trap "rm -rf $tmp" EXIT

# The Scan output is the reference the folded models must reproduce.
$TRACT_RUN --nnef-tract-core . --input-facts-from-bundle io.npz \
    run --input-from-bundle io.npz --save-outputs-npz $tmp/scan.npz \
    --assert-op-count Scan 1 --assert-op-count CumSum 0

# The caller owns the state: the running sum folds into CumSum.
$TRACT_RUN --nnef-tract-core . --input-facts-from-bundle io.npz -t force_scan_external_state \
    run --input-from-bundle io.npz --assert-output-bundle $tmp/scan.npz \
    --assert-op-count Scan 0 --assert-op-count CumSum 1

# A Scan that resets every turn folds too.
sed 's/reset_every_turn = false/reset_every_turn = true/' graph.nnef > $tmp/graph.nnef
$TRACT_RUN --nnef-tract-core $tmp --input-facts-from-bundle io.npz \
    run --input-from-bundle io.npz --assert-output-bundle $tmp/scan.npz \
    --assert-op-count Scan 0 --assert-op-count CumSum 1

# Scanning backwards is a reverse cumsum.
sed 's/("scan_input", input, 1, 1)/("scan_input", input, 1, -1)/; s/("added", "full", 1, 1)/("added", "full", 1, -1)/' \
    graph.nnef > $tmp/graph.nnef
$TRACT_RUN --nnef-tract-core $tmp --input-facts-from-bundle io.npz \
    run --input-from-bundle io.npz --save-outputs-npz $tmp/reverse.npz --assert-op-count Scan 1
$TRACT_RUN --nnef-tract-core $tmp --input-facts-from-bundle io.npz -t force_scan_external_state \
    run --input-from-bundle io.npz --assert-output-bundle $tmp/reverse.npz \
    --assert-op-count Scan 0 --assert-op-count CumSum 1
