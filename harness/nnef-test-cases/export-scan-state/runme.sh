#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# Constant-seeded state, one iteration: tract owns the carry, so the Scan stays.
$TRACT_RUN --nnef-tract-core . -t 'set_symbols(values:{"S":1})' \
    -O dump --assert-op-count Scan 1

# Same model with the state handed to the caller: the body inlines away.
$TRACT_RUN --nnef-tract-core . -t 'set_symbols(values:{"S":1})' -t export_scan_state \
    -O dump --assert-op-count Scan 0

$TRACT_RUN --nnef-tract-core . -t 'set_symbols(values:{"S":1})' -t export_scan_state \
    -O run --allow-random-input -q

# Past one iteration the scan output is the whole sequence rather than the final
# state, so the transform must decline.
$TRACT_RUN --nnef-tract-core . -t 'set_symbols(values:{"S":4})' -t export_scan_state \
    -O dump --assert-op-count Scan 1

# During warm-up (skip > 0) the Scan suppresses body execution, which an inlined
# body cannot reproduce, so the transform must decline.
tmp=`mktemp -d`
sed 's/skip *= *0/skip = 1/' graph.nnef > $tmp/graph.nnef
cp h0.dat $tmp/
$TRACT_RUN --nnef-tract-core $tmp -t 'set_symbols(values:{"S":1})' -t export_scan_state \
    -O dump --assert-op-count Scan 1
rm -rf $tmp
