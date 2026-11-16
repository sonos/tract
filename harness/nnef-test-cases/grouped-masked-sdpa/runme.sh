#!/bin/sh
set -eu
cd "$(dirname "$0")"
: "${TRACT_RUN:=cargo run -p tract-cli ${CARGO_OPTS:-} --}"
TEMP_CASE=$(mktemp -d)
trap 'rm -rf "$TEMP_CASE"' EXIT
for lengths in '1 13' '3 5'; do
    set -- $lengths
    $TRACT_RUN . --opl --set S="$1" --set T="$2" run --allow-random-input --save-outputs-npz "$TEMP_CASE/reference.npz"
    $TRACT_RUN . --opl --set S="$1" --set T="$2" -t experimental_grouped_masked_sdpa run --allow-random-input --assert-output-bundle "$TEMP_CASE/reference.npz" --approx approximate --assert-op-count SDPA 1
    if [ "${TEST_METAL:-0}" = 1 ]; then
        $TRACT_RUN . --opl --set S="$1" --set T="$2" -t experimental_grouped_masked_sdpa --metal run --allow-random-input --assert-output-bundle "$TEMP_CASE/reference.npz" --approx approximate --assert-op-count MetalMlxSdpa 1
    fi
done

mkdir "$TEMP_CASE/small"
sed 's/256.0/1.0/g' graph.nnef > "$TEMP_CASE/small/graph.nnef"
$TRACT_RUN "$TEMP_CASE/small" --opl --set S=3 --set T=5 run --allow-random-input --save-outputs-npz "$TEMP_CASE/small.npz"
$TRACT_RUN "$TEMP_CASE/small" --opl --set S=3 --set T=5 -t experimental_grouped_masked_sdpa run --allow-random-input --assert-output-bundle "$TEMP_CASE/small.npz" --approx approximate --assert-op-count SDPA 1
if [ "${TEST_METAL:-0}" = 1 ]; then
    $TRACT_RUN "$TEMP_CASE/small" --opl --set S=3 --set T=5 -t experimental_grouped_masked_sdpa --metal run --allow-random-input --assert-output-bundle "$TEMP_CASE/small.npz" --approx approximate --assert-op-count MetalMlxSdpa 1
fi
