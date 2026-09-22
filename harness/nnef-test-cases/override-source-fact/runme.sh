#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# The export pins the input to one row.
$TRACT_RUN . dump -q --assert-output-fact 1,1,f32

# --override-fact widens it, and the new fact reaches the ops downstream: a
# source carries its fact in the op, so overriding the outlet fact alone would
# leave the graph as it was.
$TRACT_RUN . --override-fact 'bias:B,1,f32' dump -q --assert-output-fact B,1,f32

# And the widened model runs.
$TRACT_RUN . --override-fact 'bias:B,1,f32' --set B=3 run -q -R \
    --assert-output-fact 3,1,f32

# Only a source's fact can be overridden: every other op derives its own, so
# the override is refused rather than ignored.
if $TRACT_RUN . --override-fact 'output:B,1,f32' dump -q > /dev/null 2>&1
then
    echo "overriding the fact of a computed node should have failed" >&2
    exit 1
fi
