#!/bin/bash
WT=${WORK}; M=${MODELS}; D="$WT/wt-cli-0.2.4"
rm -f "$D/Cargo.lock"
cd "$D/cli" || exit 1
echo "=== BUILD cli/0.2.4 rustc 1.41, 2019 index, ssl-ablated ==="
CARGO_TARGET_DIR="$WT/tgt-cli-0.2.4" timeout 2400 rustup run 1.41.0 cargo build --release 2>&1 | tail -6
echo "BUILD rc=${PIPESTATUS[0]}"
BIN="$WT/tgt-cli-0.2.4/release/tract"
if [ -x "$BIN" ]; then
  for mv in "inceptionv3:inception_v3_2016_08_28_frozen.pb:1x299x299x3xf32" "heysnips_pass:hey_snips_v4_model17.pb:200x20xf32"; do
    n=${mv%%:*}; rest=${mv#*:}; f=${rest%%:*}; sh=${rest#*:}
    for rnd in "" "--allow-random-input"; do
      out=$(timeout 300 "$BIN" "$M/$f" -i "$sh" $rnd -O bench 2>&1)
      ms=$(echo "$out" | grep -oE '[0-9.]+ ms/i' | head -1 | grep -oE '[0-9.]+')
      [ -n "$ms" ] && { echo "  $n -> $ms ms/i"; echo "2019-01,cli/0.2.4,$n,$ms" >> "$WT/curve.csv"; break; }
    done
    [ -z "$ms" ] && echo "  $n -> FAIL ($(echo "$out"|grep -iE 'error|Unmatched'|head -1|cut -c1-70))"
  done
fi
echo "BUILD2019 DONE"
