#!/bin/sh
set -eu
cd "$(dirname "$0")"
: "${TRACT_RUN:=cargo run -p tract-cli ${CARGO_OPTS:-} --}"
TEMP_CASE=$(mktemp -d)
trap 'rm -rf "$TEMP_CASE"' EXIT
$TRACT_RUN . --input 'x:3,i64=-7,0,8' --input 'y:3,i64=1,0,-2' run --save-outputs-npz "$TEMP_CASE/reference.npz"
if [ "${TEST_METAL:-0}" = 1 ]; then
    $TRACT_RUN . --input 'x:3,i64=-7,0,8' --input 'y:3,i64=1,0,-2' --metal run --assert-output-bundle "$TEMP_CASE/reference.npz" --assert-op-count DeviceSyncToHost 1
fi
