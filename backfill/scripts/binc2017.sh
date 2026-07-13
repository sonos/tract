#!/bin/bash
cd "${WORK}/wt-inception-2017" || exit 1
CARGO_TARGET_DIR="${WORK}/tgt-inception-2017" timeout 1800 rustup run 1.31.0 cargo build --release --example inceptionv3 >/dev/null 2>&1
echo "BUILD rc=$?"
BIN="${WORK}/tgt-inception-2017/release/examples/inceptionv3"
for i in 1 2 3 4; do [ -x "$BIN" ] && "$BIN" ${MODELS}/inception_v3_2016_08_28_frozen.pb 2>/dev/null; done
echo BINC2017 DONE
