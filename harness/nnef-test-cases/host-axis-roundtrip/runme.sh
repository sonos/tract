#!/bin/sh
set -eu
cd "$(dirname "$0")"
: "${TRACT_RUN:=cargo run -p tract-cli ${CARGO_OPTS:-} --}"
TEMP_CASE=$(mktemp -d)
trap 'rm -rf "$TEMP_CASE"' EXIT
$TRACT_RUN . --input 'x:3,i64=-7,0,8' --input 'y:3,i64=1,0,-2' run --save-outputs-npz "$TEMP_CASE/reference.npz"
for rt in ${TRACT_RUNTIMES:-}; do
    case "$rt" in
        --metal) assert="--assert-op-count DeviceSyncToHost 1" ;;
        --cuda|--runtime=wgpu) assert= ;;
        *) continue ;;
    esac
    $TRACT_RUN . --input 'x:3,i64=-7,0,8' --input 'y:3,i64=1,0,-2' $rt run --assert-output-bundle "$TEMP_CASE/reference.npz" $assert
done
