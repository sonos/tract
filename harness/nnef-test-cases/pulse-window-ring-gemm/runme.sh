#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# Pulses of 2 and 3 positions, the latter not dividing the window.
for rt in "" $TRACT_RUNTIMES
do
    for pulse in 2 3
    do
        $TRACT_RUN --nnef-tract-core --nnef-tract-transformers . --pulse "T=$pulse" $rt run \
            --approx approximate --input-from-bundle io.npz --assert-output-bundle io.npz
    done
done

# Both products read their window as a ring, not as a copy of it.
for rt in $TRACT_RUNTIMES
do
    case "$rt" in
        --metal) asserts="--assert-op-count MetalRingGemm 2";;
        *) continue;;
    esac
    $TRACT_RUN --nnef-tract-core --nnef-tract-transformers . --pulse "T=2" $rt dump -q $asserts
done
