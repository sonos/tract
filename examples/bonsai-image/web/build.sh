#!/bin/sh
# Builds the deployable site into dist/: the wasm, the page, the manifest and build.json.
#
# The page fetches the model from Hugging Face at the revisions site/manifest.json pins.
#
#   MODELS_DIR=<dir>   serve the three NNEF archives (and tokenizer.json, if there) from a local
#                      directory instead, linked into dist/models
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

cargo build --release -p bonsai-web --lib --target wasm32-unknown-unknown

rm -rf dist
mkdir -p dist/pkg
cp site/* dist/
wasm-bindgen --target web --out-dir dist/pkg ${CARGO_TARGET_DIR:-$ROOT/target}/wasm32-unknown-unknown/release/bonsai_web.wasm

if [ -n "$MODELS_DIR" ]
then
    ln -s `realpath $MODELS_DIR` dist/models
    python3 - dist/manifest.json "$MODELS_DIR" <<'PY'
import json, os, sys
path, root = sys.argv[1:]
manifest = json.load(open(path))
for f in manifest["files"]:
    f["url"] = f"models/{f['name']}"
    f["size"] = os.path.getsize(f"{root}/{f['name']}")
if os.path.exists(f"{root}/tokenizer.json"):
    manifest["tokenizer"] = "models/tokenizer.json"
json.dump(manifest, open(path, "w"), indent=2)
PY
fi

commit=`git rev-parse HEAD`
version=`sed -n 's/^version = "\(.*\)"$/\1/p' $ROOT/api/rs/Cargo.toml`
dirty=false
git diff --quiet HEAD -- $ROOT || dirty=true
printf '{"version":"%s","commit":"%s","dirty":%s,"date":"%s"}\n' $version $commit $dirty `date -u +%Y-%m-%d` > dist/build.json
echo "dist/ ready: $(du -shL dist/pkg | cut -f1) of wasm, commit ${commit%${commit#?????????}}$($dirty && echo +dirty)"
