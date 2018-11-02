#!/bin/sh

CRATE=$1
VERSION=$2
CRATES="core tensorflow onnx cli"

if [ -z "$VERSION" ]
then
    echo "Usage: $0 <crate> <version>" 
    echo crates order is: $CRATES
    exit 1
fi

set -ex

if [ "$CRATE" = "all" ]
then
    for c in $CRATES
    do
        $0 $c $VERSION
    done
    exit 0
fi

# set_version cargo-dinghy/Cargo.toml 0.3.0
set_version() {
    FILE=$1
    VERSION=$2
    sed -i.back "s/^version *= *\".*\"/version = \"$2\"/" $FILE
    sed -i.back "s/^\(tract-[^ =]*\).*/\\1 = \"$2\"/" $FILE
}

set_version $CRATE/Cargo.toml $VERSION
(cd $CRATE ; cargo publish --dry-run --allow-dirty)

git commit . -m "release $CRATE/$VERSION"
git tag "$CRATE/$VERSION"
git push --tags

(cd $CRATE ; cargo publish)
