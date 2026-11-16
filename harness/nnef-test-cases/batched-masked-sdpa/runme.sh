#!/bin/sh
set -eu
cd "$(dirname "$0")"
: "${TRACT_RUN:=cargo run -p tract-cli ${CARGO_OPTS:-} --}"
TEMP_CASE=$(mktemp -d)
trap 'rm -rf "$TEMP_CASE"' EXIT
mkdir "$TEMP_CASE/f32" "$TEMP_CASE/case"
sed "s/'f16'/'f32'/g" graph.nnef > "$TEMP_CASE/f32/graph.nnef"
for model in . "$TEMP_CASE/f32"; do
    for config in '1 4 1 1 13' '2 1 1 1 13' '2 4 1 1 13' '2 4 1 3 13' '2 4 3 3 13' '2 4 1 1 4096' '2 4 9 9 17'; do
        set -- $config
        sed "s/\[1, 4, M, T\]/[1, $2, M, T]/" "$model/graph.nnef" > "$TEMP_CASE/case/graph.nnef"
        $TRACT_RUN "$TEMP_CASE/case" --opl --set B="$1" --set M="$3" --set S="$4" --set T="$5" run --allow-random-input --save-outputs-npz "$TEMP_CASE/reference.npz"
        if [ "${TEST_METAL:-0}" = 1 ]; then
            $TRACT_RUN "$TEMP_CASE/case" --opl --set B="$1" --set M="$3" --set S="$4" --set T="$5" --metal run --allow-random-input --assert-output-bundle "$TEMP_CASE/reference.npz" --approx approximate --assert-op-count MetalMlxSdpa 1
        fi
    done
done
