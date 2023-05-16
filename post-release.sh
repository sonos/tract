#!/bin/sh

VERSION=$1
CRATES="data linalg core nnef nnef/nnef-resources pulse-opl pulse hir tensorflow onnx-opl onnx libcli cli ffi"

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

for f in `find * -mindepth 1 -a -name Cargo.toml`
do
    back=$(echo $(dirname $f) | sed 's/[^\/]\+/../g')
    tomato set package.version $VERSION $f > /dev/null
    for dep in $CRATES
    do
        dep=$(basename $dep)
        if tomato get dependencies.tract-$dep.version $f | grep -F . > /dev/null
        then
            tomato set dependencies.tract-$dep.version "=$VERSION" $f > /dev/null
            tomato set dependencies.tract-$dep.path $back/$dep $f > /dev/null
        fi
    done
done

git commit . -m "post-release $VERSION"
