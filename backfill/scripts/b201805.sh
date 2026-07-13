#!/bin/bash
cd "${WORK}/wt-2018-05/cli" || exit 1
echo "=== BUILD 1b62f8e80 (2018-05-21) rustc 1.31, 2018-05 index, conform ablated ==="
CARGO_TARGET_DIR="${WORK}/tgt-2018-05" timeout 1800 rustup run 1.31.0 cargo build --release 2>&1 | tail -8
echo "BUILD rc=${PIPESTATUS[0]}"
BIN="${WORK}/tgt-2018-05/release/cli"
[ -x "$BIN" ] || BIN="${WORK}/tgt-2018-05/release/tract"
[ -x "$BIN" ] && { echo "=== profile inceptionv3 (random -s) ==="; timeout 300 "$BIN" ${MODELS}/inception_v3_2016_08_28_frozen.pb -s 1x299x299x3xf32 profile 2>&1 | grep -iE 'real|ms/i|error|precursor|Entire|pass' | head -5; }
echo B201805 DONE
