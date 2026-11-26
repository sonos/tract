#!/bin/sh
# Builds the deployable site into dist/: the wasm, the page and build.json. The page downloads
# the MediaPipe hand models from Google itself.
#
# Needs the wasm32-unknown-unknown target and a wasm-bindgen CLI of the version Cargo.lock pins.

set -e
cd `dirname $0`
ROOT=`git rev-parse --show-toplevel`

want=`grep -A1 '^name = "wasm-bindgen"$' $ROOT/Cargo.lock | sed -n 's/^version = "\(.*\)"/\1/p'`
have=`wasm-bindgen --version | cut -d' ' -f2`
if [ "$want" != "$have" ]
then
    echo "wasm-bindgen $have found, Cargo.lock wants $want: cargo install wasm-bindgen-cli --version $want" >&2
    exit 1
fi

# The convolutions run on tract's simd128 kernels; without the feature they fall back to scalar code.
RUSTFLAGS="$RUSTFLAGS -C target-feature=+simd128" \
    cargo build --release -p bezier-piano-web --lib --target wasm32-unknown-unknown

rm -rf dist
mkdir -p dist/pkg
cp site/*.html site/*.js site/*.mjs dist/
rm -f dist/*.test.mjs
wasm-bindgen --target web --out-dir dist/pkg ${CARGO_TARGET_DIR:-$ROOT/target}/wasm32-unknown-unknown/release/bezier_piano_web.wasm

commit=`git rev-parse HEAD`
version=`sed -n 's/^version = "\(.*\)"$/\1/p' $ROOT/api/rs/Cargo.toml`
dirty=false
git diff --quiet HEAD -- $ROOT || dirty=true
printf '{"version":"%s","commit":"%s","dirty":%s,"date":"%s"}\n' $version $commit $dirty `date -u +%Y-%m-%d` > dist/build.json
echo "dist/ ready: $(du -sh dist | cut -f1), commit ${commit%${commit#?????????}}$($dirty && echo +dirty)"
