#!/bin/bash

set -e

git pull # make sure we are in sync
git push

cargo install tomato-toml

CRATE=$1
VERSION=$2
CRATES="data linalg core nnef nnef/nnef-resources pulse-opl pulse hir tensorflow onnx-opl onnx kaldi libcli ffi cli"

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

tomato set package.version $VERSION $CRATE/Cargo.toml
(cd $CRATE ; cargo publish --allow-dirty)

for manifest in `find * -mindepth 1 -a -name Cargo.toml`
do
    crate=$(basename $CRATE)
    if tomato get dependencies.tract-$crate.version $manifest | grep -F .
    then
        tomato set "dependencies.tract-$crate.version" "=$VERSION" $manifest
    fi
done

cargo update

if [ "$CRATE" = "cli" ]
then
    git commit -m "release $VERSION" .
    git tag -f v"$VERSION"
    git push -f --tags
fi
