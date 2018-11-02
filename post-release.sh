#!/bin/sh

VERSION=$1
CRATES="core tensorflow onnx cli"

if [ -z "$VERSION" ]
then
    echo "Usage: $0 <version>" 
    exit 1
fi

# set_version cargo-dinghy/Cargo.toml 0.3.0
set_version() {
    FILE=$1
    VERSION=$2
    sed -i.back "s/^version *= *\".*\"/version = \"$2\"/" $FILE
    sed -i.back "s/^tract-\([^ =]*\).*/tract-\\1 = { path = \"..\/\\1\" }/" $FILE
}

set -ex

for c in $CRATES
do
    set_version $c/Cargo.toml $VERSION
done

git commit . -m "post-release $VERSION"
