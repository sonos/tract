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
        symbols="--set B=$1 --set M=$3 --set S=$4 --set T=$5"
        $TRACT_RUN "$TEMP_CASE/case" --opl $symbols run --allow-random-input --save-outputs-npz "$TEMP_CASE/reference.npz"
        for rt in ${TRACT_RUNTIMES:-}; do
            case "$rt" in
                --metal) assert="--assert-op-count MetalMlxSdpa 1" ;;
                --cuda) assert="--assert-op-count CudaFlashAttention 1" ;;
                --runtime=wgpu) assert= ;;
                *) continue ;;
            esac
            $TRACT_RUN "$TEMP_CASE/case" --opl $symbols $rt run --allow-random-input --assert-output-bundle "$TEMP_CASE/reference.npz" --approx approximate $assert
        done
    done
done
