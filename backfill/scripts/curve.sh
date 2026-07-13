#!/bin/bash
# Build a cross-era evaltime curve via the crates.io-index time-machine.
WT=${WORK}
M=${MODELS}
CIDX=${CIDX}
CSV=$WT/curve.csv
mkdir -p "$CIDX"
[ -f "$CSV" ] || echo "date,tag,model,ms_per_iter" > "$CSV"
cd ${TRACT} || exit 1

# tag | snapshot-branch | until-date(tag date) | rustc | dirname
ANCHORS=(
  "cli/0.2.4|snapshot-2019-10-17|2019-01-22|1.41.0|cli-0.2.4|2019-01"
  "0.15.0|snapshot-2021-07-02|2021-06-24|1.56.0|0.15.0|2021-06"
  "0.17.0|snapshot-2022-07-06|2022-06-13|1.65.0|0.17.0|2022-06"
  "0.20.0|snapshot-2023-06-30|2023-04-25|1.68.0|0.20.0|2023-04"
)

materialize_index() { # $1 branch $2 until -> echoes registry path
  local br=$1 until=$2
  local sha
  sha=$(gh api "repos/rust-lang/crates.io-index-archive/commits?sha=${br}&until=${until}T23:59:59Z&per_page=1" -q '.[0].sha' 2>/dev/null)
  [ -z "$sha" ] && { echo "NOSHA"; return; }
  local dir="$CIDX/$sha"
  if [ ! -d "$dir/.git" ]; then
    gh api "repos/rust-lang/crates.io-index-archive/tarball/$sha" > "$CIDX/$sha.tgz" 2>/dev/null
    rm -rf "$dir"; mkdir -p "$dir"
    tar xzf "$CIDX/$sha.tgz" -C "$dir" --strip-components=1 2>/dev/null
    ( cd "$dir" && git init -q && git -c user.email=b@b -c user.name=b add -A && git -c user.email=b@b -c user.name=b commit -qm snap ) >/dev/null 2>&1
  fi
  echo "$dir"
}

run_model() { # $1 tag $2 date $3 name $4 file  rest: preargs-comma "::" preargs-x
  local tag=$1 date=$2 name=$3 file=$4 bin="$5"; shift 5
  local ca="$1" xa="$2"
  local out ms
  for pa in "$ca" "$xa"; do
    for rnd in "" "--allow-random-input"; do
      out=$(timeout 300 "$bin" "$M/$file" $pa $rnd -O bench 2>&1)
      ms=$(echo "$out" | grep -oE '[0-9.]+ ms/i' | head -1 | grep -oE '[0-9.]+')
      [ -n "$ms" ] && { echo "$date,$tag,$name,$ms" >> "$CSV"; echo "  $tag $name -> $ms ms/i"; return; }
    done
  done
  echo "$date,$tag,$name,FAIL" >> "$CSV"; echo "  $tag $name -> FAIL"
}

for a in "${ANCHORS[@]}"; do
  IFS='|' read -r tag br until rustc dir date <<< "$a"
  echo "===== $tag ($date) rustc $rustc index@$br ====="
  rustup toolchain install "$rustc" --profile minimal >/dev/null 2>&1
  reg=$(materialize_index "$br" "$until")
  [ "$reg" = "NOSHA" ] && { echo "$tag: no index sha"; continue; }
  echo "  index registry: $reg"
  git worktree remove --force "$WT/wt-$dir" >/dev/null 2>&1; rm -rf "$WT/wt-$dir" "$WT/tgt-$dir" >/dev/null 2>&1
  git worktree prune >/dev/null 2>&1
  git worktree add -f --detach "$WT/wt-$dir" "$tag" >/dev/null 2>&1 || { echo "$tag worktree FAIL"; continue; }
  mkdir -p "$WT/wt-$dir/.cargo"
  cat > "$WT/wt-$dir/.cargo/config.toml" <<CFG
[source.crates-io]
replace-with = "snap"
[source.snap]
registry = "file://$reg"
CFG
  rm -f "$WT/wt-$dir/Cargo.lock"
  ( cd "$WT/wt-$dir/cli" && CARGO_TARGET_DIR="$WT/tgt-$dir" timeout 2400 rustup run "$rustc" cargo build --release ) > "$WT/cbuild-$dir.log" 2>&1
  bin="$WT/tgt-$dir/release/tract"
  if [ ! -x "$bin" ]; then echo "$tag BUILD FAIL ($(grep -m1 -E '^error' "$WT/cbuild-$dir.log" | cut -c1-70))"; echo "$date,$tag,BUILD,FAIL" >> "$CSV"; continue; fi
  echo "$tag build OK"
  run_model "$tag" "$date" inceptionv3   "inception_v3_2016_08_28_frozen.pb" "$bin" "-i 1,299,299,3,f32" "-i 1x299x299x3xf32"
  run_model "$tag" "$date" heysnips_pass "hey_snips_v4_model17.pb"           "$bin" "-i 200,20,f32"       "-i 200x20xf32"
  run_model "$tag" "$date" tdnn15M_2600  "en_tdnn_15M.onnx"                  "$bin" "--output-node output -i 264,40" "--output-node output -i 264x40"
done
echo "CURVE DONE"
