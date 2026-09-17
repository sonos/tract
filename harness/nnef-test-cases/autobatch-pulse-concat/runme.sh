#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

PULSE='-t pulse(symbol:Some("T"),pulse:"4")'

# io.npz was minted by tract itself, and holds what it made and what it got:
# `--set B=1 --set T=16 run -R --save-steps io.npz` on the unpulsified model.

# The leading context is six frames against a pulse of four, so it spans two
# pulses: the op has to still be writing it on the second turn, which is what
# makes the position it keeps observable at all.
$TRACT_RUN --nnef-tract-core --set B=1 --set T=16 . run --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz
$TRACT_RUN --nnef-tract-core . $PULSE --set B=1 run --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz
$TRACT_RUN --nnef-tract-core . $PULSE --set B=1 dump -q \
    --assert-op-count PulsedSameAxisConcat 1 --assert-op-count Delay 1

# Four streams on four lanes of one state, each against the same stream run
# alone. The position is per stream, so a turn seating four of them advances
# four positions: on one shared counter the second turn is already past the
# context and stops writing it, which is a diff here and not a shape error.
TRACT_TURN_LINGER_US=200000 $TRACT_RUN --nnef-tract-core . $PULSE \
    --autobatch-sessions 4 --hint B=4 run --streams 4 --approx exact \
    --assert-occupancy 2.5 --input-from-bundle io.npz
