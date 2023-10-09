#!/bin/bash

set -e

git pull # make sure we are in sync
git push

cargo install tomato-toml

CRATE_PATH=$1
VERSION=$2
ALL_CRATES_PATH="data linalg core nnef nnef/nnef-resources pulse-opl pulse extra hir tflite tensorflow onnx-opl onnx libcli api api/rs api/ffi api/proxy/sys api/proxy cli"

if [ -z "$VERSION" ]
then
    echo "Usage: $0 <crate> <version>" 
    echo crates order is: $ALL_CRATES_PATH
    exit 1
fi

set -ex

if [ "$CRATE_PATH" = "all" ]
then
    for c in $ALL_CRATES_PATH
    do
        $0 $c $VERSION
    done
    exit 0
fi

crate=$(tomato get package.name $CRATE_PATH/Cargo.toml)
tomato set package.version $VERSION $CRATE_PATH/Cargo.toml
cargo publish --allow-dirty -p $crate

for other_cargo_toml in `find . -name Cargo.toml \!  -path "./target/*" \! -path "./issue*"`
do
    if tomato get dependencies.$crate $other_cargo_toml | grep .
    then
        tomato set "dependencies.$crate.version" "=$VERSION" $other_cargo_toml
    fi
    if tomato get dev-dependencies.$crate $other_cargo_toml | grep .
    then
        tomato set "dev-dependencies.$crate.version" "=$VERSION" $other_cargo_toml
    fi
done

cargo update

if [ "$CRATE_PATH" = "cli" ]
then
    git commit -m "release $VERSION" .
    git tag -f v"$VERSION"
    git push -f --tags
fi
