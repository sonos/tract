#!/bin/sh
set -eu
cd "$(dirname "$0")"
: "${TRACT_RUN:=cargo run -p tract-cli ${CARGO_OPTS:-} --}"
TEMP_CASE=$(mktemp -d)
trap 'rm -rf "$TEMP_CASE"' EXIT
mkdir "$TEMP_CASE/reference"
cp reference.nnef "$TEMP_CASE/reference/graph.nnef"
for lengths in '1 13' '3 5' '9 17' '1 4096'; do
    set -- $lengths
    $TRACT_RUN "$TEMP_CASE/reference" --opl --input 'sinks:4,f32=-20,-2,2,20' --set S="$1" --set T="$2" run --allow-random-input --save-outputs-npz "$TEMP_CASE/reference.npz"
    $TRACT_RUN . --opl --input 'sinks:4,f32=-20,-2,2,20' --set S="$1" --set T="$2" --nnef-cycle run --allow-random-input --assert-output-bundle "$TEMP_CASE/reference.npz" --approx approximate
    if [ "${TEST_METAL:-0}" = 1 ]; then
        $TRACT_RUN . --opl --input 'sinks:4,f32=-20,-2,2,20' --set S="$1" --set T="$2" --metal run --allow-random-input --assert-output-bundle "$TEMP_CASE/reference.npz" --approx approximate --assert-op-count MetalMlxSdpa 1
    fi
done

$TRACT_RUN "$TEMP_CASE/reference" --opl --set S=1 --set T=1 --input 'mask:1,1,1,1,f32=-inf' run --allow-random-input --save-outputs-npz "$TEMP_CASE/masked.npz"
$TRACT_RUN . --opl --set S=1 --set T=1 --input 'mask:1,1,1,1,f32=-inf' run --allow-random-input --assert-output-bundle "$TEMP_CASE/masked.npz" --approx approximate
if [ "${TEST_METAL:-0}" = 1 ]; then
    $TRACT_RUN . --opl --set S=1 --set T=1 --input 'mask:1,1,1,1,f32=-inf' --metal run --allow-random-input --assert-output-bundle "$TEMP_CASE/masked.npz" --approx approximate --assert-op-count MetalMlxSdpa 1
fi
mkdir "$TEMP_CASE/f32" "$TEMP_CASE/f32-reference"
sed "s/'f16'/'f32'/g" graph.nnef > "$TEMP_CASE/f32/graph.nnef"
sed "s/'f16'/'f32'/g" reference.nnef > "$TEMP_CASE/f32-reference/graph.nnef"
for lengths in '1 13' '9 17'; do
    set -- $lengths
    $TRACT_RUN "$TEMP_CASE/f32-reference" --opl --set S="$1" --set T="$2" --input 'sinks:4,f32=-20,-2,2,20' run --allow-random-input --save-outputs-npz "$TEMP_CASE/f32.npz"
    $TRACT_RUN "$TEMP_CASE/f32" --opl --set S="$1" --set T="$2" --input 'sinks:4,f32=-20,-2,2,20' run --allow-random-input --assert-output-bundle "$TEMP_CASE/f32.npz" --approx approximate
    if [ "${TEST_METAL:-0}" = 1 ]; then
        $TRACT_RUN "$TEMP_CASE/f32" --opl --set S="$1" --set T="$2" --input 'sinks:4,f32=-20,-2,2,20' --metal run --allow-random-input --assert-output-bundle "$TEMP_CASE/f32.npz" --approx approximate --assert-op-count MetalMlxSdpa 1
    fi
done

mkdir "$TEMP_CASE/batched" "$TEMP_CASE/batched-f32"
sed -e 's/\[1, 4, S, 64\]/[2, 4, S, 64]/' -e 's/\[1, 2, T, 64\]/[2, 2, T, 64]/g' -e 's/\[1, 1, S, T\]/[1, 4, S, T]/' graph.nnef > "$TEMP_CASE/batched/graph.nnef"
sed "s/'f16'/'f32'/g" "$TEMP_CASE/batched/graph.nnef" > "$TEMP_CASE/batched-f32/graph.nnef"
for model in "$TEMP_CASE/batched" "$TEMP_CASE/batched-f32"; do
    for lengths in '1 13' '1 4096' '9 17'; do
        set -- $lengths
        $TRACT_RUN "$model" --opl --set S="$1" --set T="$2" run --allow-random-input --save-outputs-npz "$TEMP_CASE/batched.npz"
        if [ "${TEST_METAL:-0}" = 1 ]; then
            $TRACT_RUN "$model" --opl --set S="$1" --set T="$2" --metal run --allow-random-input --assert-output-bundle "$TEMP_CASE/batched.npz" --approx approximate --assert-op-count MetalMlxSdpa 1
        fi
    done
done

TRACT_RUN="$TRACT_RUN" TEST_SINKS=1 sh ../sdpa-special-values/runme.sh
