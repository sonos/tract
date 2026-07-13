#!/bin/bash
WT=${WORK}
M=${MODELS}
CIDX=${CIDX}
CSV=$WT/curve.csv
cd ${TRACT} || exit 1

materialize_index() {
  local br=$1 until=$2 sha dir
  sha=$(gh api "repos/rust-lang/crates.io-index-archive/commits?sha=${br}&until=${until}T23:59:59Z&per_page=1" -q '.[0].sha' 2>/dev/null)
  [ -z "$sha" ] && { echo NOSHA; return; }
  dir="$CIDX/$sha"
  if [ ! -d "$dir/.git" ]; then
    gh api "repos/rust-lang/crates.io-index-archive/tarball/$sha" > "$CIDX/$sha.tgz" 2>/dev/null
    rm -rf "$dir"; mkdir -p "$dir"; tar xzf "$CIDX/$sha.tgz" -C "$dir" --strip-components=1 2>/dev/null
    ( cd "$dir" && git init -q && git -c user.email=b@b -c user.name=b add -A && git -c user.email=b@b -c user.name=b commit -qm snap ) >/dev/null 2>&1
  fi
  echo "$dir"
}
run_model() {
  local tag=$1 date=$2 name=$3 file=$4 bin=$5 ca=$6 xa=$7 out ms
  for pa in "$ca" "$xa"; do for rnd in "" "--allow-random-input"; do
    out=$(timeout 300 "$bin" "$M/$file" $pa $rnd -O bench 2>&1)
    ms=$(echo "$out" | grep -oE '[0-9.]+ ms/i' | head -1 | grep -oE '[0-9.]+')
    [ -n "$ms" ] && { echo "$date,$tag,$name,$ms" >> "$CSV"; echo "  $tag $name -> $ms"; return; }
  done; done
  echo "$date,$tag,$name,FAIL" >> "$CSV"; echo "  $tag $name -> FAIL"
}

ANCHORS=(
  "0.19.12|snapshot-2023-06-30|2023-04-20|1.68.0|0.19.12e|2023-04"
  "0.21.7|snapshot-2024-11-27|2024-09-23|1.81.0|0.21.7e|2024-09"
)
for a in "${ANCHORS[@]}"; do
  IFS='|' read -r tag br until rustc dir date <<< "$a"
  echo "===== $tag ($date) rustc $rustc index@$br ====="
  rustup toolchain install "$rustc" --profile minimal >/dev/null 2>&1
  reg=$(materialize_index "$br" "$until"); [ "$reg" = NOSHA ] && { echo "$tag no sha"; continue; }
  git worktree remove --force "$WT/wt-$dir" >/dev/null 2>&1; rm -rf "$WT/wt-$dir" "$WT/tgt-$dir"; git worktree prune >/dev/null 2>&1
  git worktree add -f --detach "$WT/wt-$dir" "$tag" >/dev/null 2>&1 || { echo "$tag wt FAIL"; continue; }
  mkdir -p "$WT/wt-$dir/.cargo"
  printf '[source.crates-io]\nreplace-with = "snap"\n[source.snap]\nregistry = "file://%s"\n' "$reg" > "$WT/wt-$dir/.cargo/config.toml"
  rm -f "$WT/wt-$dir/Cargo.lock"
  ( cd "$WT/wt-$dir/cli" && CARGO_TARGET_DIR="$WT/tgt-$dir" timeout 2400 rustup run "$rustc" cargo build --release ) > "$WT/cbuild-$dir.log" 2>&1
  bin="$WT/tgt-$dir/release/tract"
  [ -x "$bin" ] || { echo "$tag BUILD FAIL ($(grep -m1 -E '^error' "$WT/cbuild-$dir.log"|cut -c1-70))"; continue; }
  echo "$tag build OK"
  run_model "$tag" "$date" inceptionv3 "inception_v3_2016_08_28_frozen.pb" "$bin" "-i 1,299,299,3,f32" "-i 1x299x299x3xf32"
  run_model "$tag" "$date" heysnips_pass "hey_snips_v4_model17.pb" "$bin" "-i 200,20,f32" "-i 200x20xf32"
  run_model "$tag" "$date" tdnn15M_2600 "en_tdnn_15M.onnx" "$bin" "--output-node output -i 264,40" "--output-node output -i 264x40"
done
echo "CURVE2 DONE"
