#!/bin/sh
set -eu
cd "$(dirname "$0")"
: "${TRACT_RUN:=cargo run -p tract-cli ${CARGO_OPTS:-} --}"
TEMP_CASE=$(mktemp -d)
trap 'rm -rf "$TEMP_CASE"' EXIT
for lengths in '1 13' '9 17'; do
    set -- $lengths
    $TRACT_RUN . --opl --set S="$1" --set T="$2" run --allow-random-input --save-outputs-npz "$TEMP_CASE/reference.npz"
    if [ "${TEST_METAL:-0}" = 1 ]; then
        $TRACT_RUN . --opl --set S="$1" --set T="$2" --metal run --allow-random-input --assert-output-bundle "$TEMP_CASE/reference.npz" --approx approximate --assert-op-count MetalMlxSdpa 1
    fi
done
if [ "${TEST_METAL:-0}" = 1 ]; then
    $TRACT_RUN . --opl --metal dump -q --assert-op-count MetalMlxSdpa 1
fi

mkdir "$TEMP_CASE/batched"
sed -e 's/\[1, 4, S, 64\]/[2, 4, S, 64]/' -e 's/\[1, 2, T, 64\]/[2, 2, T, 64]/g' -e 's/\[1, 1, S, T\]/[1, 4, S, T]/' graph.nnef > "$TEMP_CASE/batched/graph.nnef"
for lengths in '1 13' '1 4096' '9 17'; do
    set -- $lengths
    $TRACT_RUN "$TEMP_CASE/batched" --opl run --set S="$1" --set T="$2" --allow-random-input --save-outputs-npz "$TEMP_CASE/batched.npz"
    if [ "${TEST_METAL:-0}" = 1 ]; then
        $TRACT_RUN "$TEMP_CASE/batched" --opl --metal run --set S="$1" --set T="$2" --allow-random-input --assert-output-bundle "$TEMP_CASE/batched.npz" --approx approximate --assert-op-count MetalMlxSdpa 1
    fi
done

TRACT_RUN="$TRACT_RUN" TEST_SYMBOLIC=1 sh ../sdpa-special-values/runme.sh
