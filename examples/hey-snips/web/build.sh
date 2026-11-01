#!/bin/sh
# Builds the deployable site into dist/: the wasm, the page, the model and build.json.
#
#   GRAPHS_DIR=<dir>   take hey_snips.nnef.tar from a local directory (export.sh writes it)
#   GRAPHS_URL=<url>   where the published graph lives
#
# Needs the wasm32-unknown-unknown target and a wasm-bindgen CLI of the version Cargo.lock pins.

set -e
cd `dirname $0`
ROOT=`git rev-parse --show-toplevel`
: ${GRAPHS_URL:=https://tract-test-assets.tract.rs/hey-snips/v1}

want=`grep -A1 '^name = "wasm-bindgen"$' $ROOT/Cargo.lock | sed -n 's/^version = "\(.*\)"/\1/p'`
have=`wasm-bindgen --version | cut -d' ' -f2`
if [ "$want" != "$have" ]
then
    echo "wasm-bindgen $have found, Cargo.lock wants $want: cargo install wasm-bindgen-cli --version $want" >&2
    exit 1
fi

cargo build --release -p hey-snips-web --lib --target wasm32-unknown-unknown

rm -rf dist
mkdir -p dist/pkg dist/models
cp site/* dist/
wasm-bindgen --target web --out-dir dist/pkg ${CARGO_TARGET_DIR:-$ROOT/target}/wasm32-unknown-unknown/release/hey_snips_web.wasm

if [ -n "$GRAPHS_DIR" ]
then
    cp $GRAPHS_DIR/hey_snips.nnef.tar dist/models/
else
    curl -fsSL $GRAPHS_URL/hey_snips.nnef.tar -o dist/models/hey_snips.nnef.tar
fi

commit=`git rev-parse HEAD`
dirty=false
git diff --quiet HEAD -- $ROOT || dirty=true
printf '{"commit":"%s","dirty":%s,"date":"%s"}\n' $commit $dirty `date -u +%Y-%m-%d` > dist/build.json
echo "dist/ ready: $(du -sh dist | cut -f1), commit ${commit%${commit#?????????}}$($dirty && echo +dirty)"
