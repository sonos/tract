#!/bin/bash
WT=${WORK}
M=${MODELS}
cd "$WT/wt-0.10.10/cli" || exit 1
echo "=== BUILD 0.10.10 rustc 1.46, resolving vs 2020-08-28 index snapshot ==="
CARGO_TARGET_DIR="$WT/tgt-0.10.10" timeout 2400 rustup run 1.46.0 cargo build --release 2>&1
echo "BUILD rc=$?"
BIN="$WT/tgt-0.10.10/release/tract"
if [ -x "$BIN" ]; then
  echo "=== proc-macro2 version actually used ==="; grep -A1 'name = "proc-macro2"' "$WT/wt-0.10.10/Cargo.lock" | grep version
  echo "=== RUN inceptionv3 ==="; timeout 300 "$BIN" "$M/inception_v3_2016_08_28_frozen.pb" -i 1x299x299x3xf32 -O bench 2>&1 | tail -3
  echo "=== RUN hey_snips_v4 pass ==="; timeout 300 "$BIN" "$M/hey_snips_v4_model17.pb" -i 200x20xf32 -O bench 2>&1 | tail -3
fi
echo "BUILD010C DONE"
