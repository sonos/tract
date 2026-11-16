#!/bin/sh
set -eu
cd "$(dirname "$0")"
: "${TRACT_RUN:=cargo run -p tract-cli ${CARGO_OPTS:-} --}"
TEMP_CASE=$(mktemp -d)
trap 'rm -rf "$TEMP_CASE"' EXIT
for lengths in '1 13' '9 17'; do
    set -- $lengths
    $TRACT_RUN . --opl --input 'sinks:4,f32=-20,-2,2,20' --set S="$1" --set T="$2" run --allow-random-input --save-outputs-npz "$TEMP_CASE/pattern.npz"
    $TRACT_RUN . --opl --input 'sinks:4,f32=-20,-2,2,20' --set S="$1" --set T="$2" -t experimental_sink_sdpa run --allow-random-input --assert-output-bundle "$TEMP_CASE/pattern.npz" --approx approximate --assert-op-count SDPA 1
done

mkdir "$TEMP_CASE/negative"
sed 's/maximum = max_reduce(logits, axes = \[2\])/maximum = max_reduce(logits, axes = [1])/' graph.nnef > "$TEMP_CASE/negative/graph.nnef"
$TRACT_RUN "$TEMP_CASE/negative" --opl -t experimental_sink_sdpa dump -q --assert-op-count SDPA 0
