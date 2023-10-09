#!/bin/sh

VERSION=$1
ALL_CRATES_PATH="data linalg core nnef nnef/nnef-resources pulse-opl pulse extra hir tflite tensorflow onnx-opl onnx libcli api api/rs api/ffi api/proxy/sys api/proxy cli"

if [ `uname` = "Darwin" ]
then
    SED=gsed
else
    SED=sed
fi

if [ -z "$VERSION" ]
then
    echo "Usage: $0 <version>" 
    exit 1
fi

for path in $ALL_CRATES_PATH
do
    crate=$(tomato get package.name $path/Cargo.toml)
    tomato set package.version $VERSION $path/Cargo.toml > /dev/null
    for other_cargo_toml in `find . -name Cargo.toml \!  -path "./target/*" \! -path "./issue*"`
    do
        if tomato get dependencies.$crate $other_cargo_toml | grep -F . > /dev/null
        then
            tomato set dependencies.$crate.version "=$VERSION" $other_cargo_toml > /dev/null
        fi
        if tomato get dev-dependencies.$crate $other_cargo_toml | grep -F . > /dev/null
        then
            tomato set dev-dependencies.$crate.version "=$VERSION" $other_cargo_toml > /dev/null
        fi
    done
done

git commit . -m "post-release $VERSION"
