#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

PULSE='-t pulse(symbol:Some("T"),pulse:"2")'

# Batch, and pulsified with the lane axis pinned to the one row a stream feeds.
$TRACT_RUN --nnef-tract-core --set T=16 --set B=1 . run --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz
$TRACT_RUN --nnef-tract-core . $PULSE --set B=1 run --approx approximate \
    --input-from-bundle io.npz --assert-output-bundle io.npz
$TRACT_RUN --nnef-tract-core . $PULSE dump -q \
    --assert-op-count Delay 1 --assert-op-count PulsePad 1

# Laned, batch axis left symbolic: four streams share one state, each on a lane
# of its own, and each must get what it gets alone. The seats stagger
# themselves -- a stream feeds the turns rotated by its own index -- so a
# `Delay` or `PulsePad` two streams share shows up as a diff here. The
# reference bundle is not asserted against a laned run: the width of a turn
# decides how its sums associate, so only the same model run alone is an exact
# reference.
$TRACT_RUN --nnef-tract-core . $PULSE --lanes 4 --hint B=4 run --streams 4 \
    --approx exact --input-from-bundle io.npz

# The same, with the worker lingering for the streams that are not queued yet:
# whatever the box's scheduling, the turns are wide, so the batch axis and the
# seating are actually exercised.
TRACT_TURN_LINGER_US=100000 $TRACT_RUN --nnef-tract-core . $PULSE --lanes 4 \
    --hint B=4 run --streams 4 --approx exact --assert-occupancy 3.0 \
    --input-from-bundle io.npz
