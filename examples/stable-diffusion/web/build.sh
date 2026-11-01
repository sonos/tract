#!/bin/sh
# Builds the deployable site into dist/: the wasm, the page, the graphs and build.json.
#
#   GRAPHS_DIR=<dir>   take the graph tars from a local directory instead of R2
#   GRAPHS_URL=<url>   where the published graph tars live
#
# Needs the wasm32-unknown-unknown target and a wasm-bindgen CLI of the version Cargo.lock pins.

set -e
cd `dirname $0`
ROOT=`git rev-parse --show-toplevel`
: ${GRAPHS_URL:=https://tract-test-assets.tract.rs/sd/sd15-web/v1}

want=`grep -A1 '^name = "wasm-bindgen"$' $ROOT/Cargo.lock | sed -n 's/^version = "\(.*\)"/\1/p'`
have=`wasm-bindgen --version | cut -d' ' -f2`
if [ "$want" != "$have" ]
then
    echo "wasm-bindgen $have found, Cargo.lock wants $want: cargo install wasm-bindgen-cli --version $want" >&2
    exit 1
fi

cargo build --release -p sd15-web --lib --target wasm32-unknown-unknown

rm -rf dist
mkdir -p dist/pkg dist/models
cp site/* dist/
wasm-bindgen --target web --out-dir dist/pkg ${CARGO_TARGET_DIR:-$ROOT/target}/wasm32-unknown-unknown/release/sd15_web.wasm

for c in text_encoder unet vae_decoder
do
    if [ -n "$GRAPHS_DIR" ]
    then
        cp $GRAPHS_DIR/$c.nnef.tar dist/models/
    else
        curl -fsSL $GRAPHS_URL/$c.nnef.tar -o dist/models/$c.nnef.tar
    fi
done

commit=`git rev-parse HEAD`
dirty=false
git diff --quiet HEAD -- $ROOT || dirty=true
printf '{"commit":"%s","dirty":%s,"date":"%s"}\n' $commit $dirty `date -u +%Y-%m-%d` > dist/build.json
echo "dist/ ready: $(du -sh dist | cut -f1), commit ${commit%${commit#?????????}}$($dirty && echo +dirty)"
