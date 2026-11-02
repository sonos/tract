#!/bin/sh
# Builds the deployable site into dist/: the wasm, the page and build.json. The models are not part
# of it: the page fetches them from Hugging Face.
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

cargo build --release -p pixel-lab-web --lib --target wasm32-unknown-unknown

rm -rf dist
mkdir -p dist/pkg
cp site/*.html site/*.js site/*.mjs dist/
rm -f dist/*.test.mjs
wasm-bindgen --target web --out-dir dist/pkg ${CARGO_TARGET_DIR:-$ROOT/target}/wasm32-unknown-unknown/release/pixel_lab_web.wasm

commit=`git rev-parse HEAD`
dirty=false
git diff --quiet HEAD -- $ROOT || dirty=true
printf '{"commit":"%s","dirty":%s,"date":"%s"}\n' $commit $dirty `date -u +%Y-%m-%d` > dist/build.json
echo "dist/ ready: $(du -sh dist | cut -f1), commit ${commit%${commit#?????????}}$($dirty && echo +dirty)"
