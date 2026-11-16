#!/bin/sh
set -eu
cd "$(dirname "$0")"
: "${TRACT_RUN:=cargo run -p tract-cli ${CARGO_OPTS:-} --}"
TEMP_CASE=$(mktemp -d)
trap 'rm -rf "$TEMP_CASE"' EXIT
mkdir "$TEMP_CASE/reference"
cp reference.nnef "$TEMP_CASE/reference/graph.nnef"
$TRACT_RUN "$TEMP_CASE/reference" --opl --set S=3 --set T=13 run --allow-random-input --save-outputs-npz "$TEMP_CASE/reference.npz"
$TRACT_RUN . --opl --set S=3 --set T=13 --nnef-cycle run --allow-random-input --assert-output-bundle "$TEMP_CASE/reference.npz" --approx approximate
