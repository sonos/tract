#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# The graph names its weights; they live in weights.safetensors, outside the bundle, and are
# read from it once linked.
LINK="--link-safetensors weights.safetensors"
$TRACT_RUN model $LINK run --input-from-npz io.npz --assert-output-bundle io.npz -q
$TRACT_RUN model $LINK -O run --input-from-npz io.npz --assert-output-bundle io.npz -q

for rt in $TRACT_RUNTIMES
do
    $TRACT_RUN model $rt $LINK run --input-from-npz io.npz --assert-output-bundle io.npz \
        --approx approximate -q
done

# Unlinked, the weights have nowhere to be read from.
if $TRACT_RUN model run --input-from-npz io.npz -q 2> unlinked.log
then
    exit 1
fi
grep -q 'has no source' unlinked.log
rm -f unlinked.log
