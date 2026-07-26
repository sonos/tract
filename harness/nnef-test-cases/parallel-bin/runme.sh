#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

trap 'rm -f ref.npz' EXIT

# --allow-random-input is seeded from a fixed constant, so both runs below get
# byte-identical input without committing a bundle.
#
# Serial reference on this same binary/platform (the default executor is serial).
$TRACT_RUN . -O run --allow-random-input --save-outputs ref.npz

# par_bin only splits `a` on boundaries the kernels already tolerate, so the
# threaded result must be bit-identical to the serial one (--approx exact =
# atol=rtol=0). Comparing against a same-platform serial run keeps this
# independent of which SIMD kernel the host registers.
#
# A default build (including the CI cli-tests harness) has no multithread-mm, so
# `--threads` bails; drive this case with a multithread-mm TRACT_RUN (the plan's
# verification step does) to exercise the parallel leg. We warn rather than skip
# silently so a serial-only run is never mistaken for threaded coverage.
if $TRACT_RUN --threads 2 . dump -q >/dev/null 2>&1; then
    $TRACT_RUN --threads 8 --threading-threshold 0 . -O run \
        --allow-random-input --assert-output-bundle ref.npz --approx exact
else
    echo "WARNING: TRACT_RUN has no multithread-mm; parallel bit-identity leg NOT run." >&2
fi
