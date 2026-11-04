#!/bin/sh
# Builds the deployable site into dist/: the wasm, the page and the manifest.
#
#   MODELS_DIR=<dir>   the stage tars and tokenizer.json, linked into dist/models
#   MODELS_URL=<url>   where they are published, instead of a local directory
#
# Needs the wasm32-unknown-unknown target and a wasm-bindgen CLI of the version Cargo.lock pins.

set -e
cd `dirname $0`
ROOT=`git rev-parse --show-toplevel`
: ${MODELS_DIR:?stage tars and tokenizer.json, from export.sh}

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

if [ -n "$MODELS_URL" ]
then
    base=$MODELS_URL
else
    ln -s `realpath $MODELS_DIR` dist/models
    base=models
fi

python3 - "$MODELS_DIR" "$base" > dist/manifest.json <<'PY'
import glob, json, os, re, sys
root, base = sys.argv[1:]
parts = sorted(glob.glob(f"{root}/transformer-*.tar"))
sizes = [re.search(r"-(\d+x\d+)\.tar", p)[1] for p in parts]
names = ["text_encoder.nnef.tar"] + [
    name
    for stage in ("transformer", "vae_decoder")
    for name in [f"{stage}.weights.tar"] + [f"{stage}-{size}.tar" for size in sizes]
]
print(json.dumps({
    "tokenizer": f"{base}/tokenizer.json",
    "sizes": sizes,
    "files": [
        {"name": n, "url": f"{base}/{n}", "size": os.path.getsize(f"{root}/{n}")} for n in names
    ],
}, indent=2))
PY

commit=`git rev-parse HEAD`
dirty=false
git diff --quiet HEAD -- $ROOT || dirty=true
printf '{"commit":"%s","dirty":%s,"date":"%s"}\n' $commit $dirty `date -u +%Y-%m-%d` > dist/build.json
echo "dist/ ready: commit ${commit%${commit#?????????}}$($dirty && echo +dirty)"
