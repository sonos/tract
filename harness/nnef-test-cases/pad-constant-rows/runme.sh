#!/bin/sh

cd "$(dirname "$0")"
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

$TRACT_RUN . -O run --input-from-bundle io.npz --assert-output-bundle io.npz --approx exact
if $TRACT_RUN --threads 2 . dump -q >/dev/null 2>&1; then
    $TRACT_RUN --threads 4 . -O run --input-from-bundle io.npz --assert-output-bundle io.npz --approx exact
else
    echo "WARNING: TRACT_RUN has no multithread-mm; parallel pad leg NOT run." >&2
fi
