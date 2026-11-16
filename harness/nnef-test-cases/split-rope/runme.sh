#!/bin/sh
set -eu
cd "$(dirname "$0")"
: "${TRACT_RUN:=cargo run -p tract-cli ${CARGO_OPTS:-} --}"
TEMP_CASE=$(mktemp -d)
trap 'rm -rf "$TEMP_CASE"' EXIT
for length in 1 3 17; do
    $TRACT_RUN . --opl --set S="$length" run --allow-random-input --save-outputs-npz "$TEMP_CASE/reference.npz"
    $TRACT_RUN . --opl --set S="$length" -t experimental_split_rope run --allow-random-input --assert-output-bundle "$TEMP_CASE/reference.npz" --approx approximate --assert-op-count ApplyRope 1
    if [ "${TEST_METAL:-0}" = 1 ]; then
        $TRACT_RUN . --opl --set S="$length" -t experimental_split_rope --metal run --allow-random-input --assert-output-bundle "$TEMP_CASE/reference.npz" --approx approximate --assert-op-count MetalApplyRope 1
    fi
done
mkdir "$TEMP_CASE/negative"
sed 's/hi = add(bc, ass)/hi = sub(bc, ass)/' graph.nnef > "$TEMP_CASE/negative/graph.nnef"
$TRACT_RUN "$TEMP_CASE/negative" --opl -t experimental_split_rope dump -q --assert-op-count ApplyRope 0

mkdir "$TEMP_CASE/f32"
sed "s/'f16'/'f32'/g" graph.nnef > "$TEMP_CASE/f32/graph.nnef"
$TRACT_RUN "$TEMP_CASE/f32" --opl --set S=3 run --allow-random-input --save-outputs-npz "$TEMP_CASE/f32.npz"
$TRACT_RUN "$TEMP_CASE/f32" --opl --set S=3 -t experimental_split_rope run --allow-random-input --assert-output-bundle "$TEMP_CASE/f32.npz" --approx approximate --assert-op-count ApplyRope 1
if [ "${TEST_METAL:-0}" = 1 ]; then
    $TRACT_RUN "$TEMP_CASE/f32" --opl --set S=3 -t experimental_split_rope --metal run --allow-random-input --assert-output-bundle "$TEMP_CASE/f32.npz" --approx approximate --assert-op-count MetalApplyRope 1
fi
