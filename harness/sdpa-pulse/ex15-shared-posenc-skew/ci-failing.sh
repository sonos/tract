#!/bin/sh
cd "$(dirname "$0")"
set -ex
: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}
python3 gen-inputs.py

$TRACT_RUN --nnef-tract-core --nnef-tract-transformers . \
    run --input-from-bundle io.npz --assert-output-bundle io.npz --approx very

$TRACT_RUN --nnef-tract-core --nnef-tract-transformers . \
    -t 'pulse(symbol: Some("S"), pulse: "4")' \
    run --input-from-bundle io.npz --assert-output-bundle io.npz --approx very
