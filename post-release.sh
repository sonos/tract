#!/bin/sh

VERSION=$1
ALL_CRATES_PATH="data linalg core nnef nnef/nnef-resources pulse-opl pulse hir tflite tensorflow onnx-opl onnx libcli api api/rs api/proxy cli"

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

for other_path in `find . -name Cargo.toml`
do
    crate=$(tomato get package.name $path/Cargo.toml)
    tomato set package.version $VERSION $path/Cargo.toml > /dev/null
    for other_path in $ALL_CRATES_PATH
    do
        if tomato get dependencies.$crate.version $other_path/Cargo.toml | grep -F . > /dev/null
        then
            tomato set dependencies.$crate.version "=$VERSION" $other_path/Cargo.toml > /dev/null
        fi
    done
done

git commit . -m "post-release $VERSION"
