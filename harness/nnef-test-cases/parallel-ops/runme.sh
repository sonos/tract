#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# Serial reference on this same binary/platform (the default executor is serial).
$TRACT_RUN . -O run --input-from-bundle io.npz --save-outputs ref.npz

# The row-parallel path only exists in a multithread-mm build. When TRACT_RUN
# has it, re-run forced fully parallel (--threading-threshold 0) and require the
# result to be bit-identical to the serial reference (--approx exact =
# atol=rtol=0): Part 1 only splits the outer/row axis, so any leak into a
# per-row reduction shows up here. Comparing against a same-platform serial run
# keeps this independent of the SIMD softmax/exp approximations, which differ
# across architectures.
#
# A default build (including the CI cli-tests harness) has no multithread-mm, so
# `--threads` bails; drive this case with a multithread-mm TRACT_RUN (the plan's
# verification step does) to exercise the parallel leg. We warn rather than skip
# silently so a serial-only run is never mistaken for threaded coverage.
if $TRACT_RUN --threads 2 . dump -q >/dev/null 2>&1; then
    $TRACT_RUN --threads 8 --threading-threshold 0 . -O run \
        --input-from-bundle io.npz --assert-output-bundle ref.npz --approx exact
else
    echo "WARNING: TRACT_RUN has no multithread-mm; parallel bit-identity leg NOT run." >&2
fi

rm -f ref.npz
