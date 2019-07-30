#!/bin/sh

CRATE=$1
VERSION=$2
CRATES="linalg core tensorflow onnx kaldi cli"

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
    for dep in `grep "^tract-" $FILE | cut -d " " -f 1`
    do
        cargo add --manifest-path $FILE $dep@$VERSION
    done
}

cargo update

set_version $CRATE/Cargo.toml $VERSION
(cd $CRATE ; cargo publish --dry-run --allow-dirty)

git commit . -m "release $CRATE/$VERSION"
git tag -f "$CRATE/$VERSION"
git push -f --tags

(cd $CRATE ; cargo publish --allow-dirty)
cargo update
sleep 5
