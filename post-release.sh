#!/bin/sh

VERSION=$1
CRATES="data linalg core nnef pulse-opl pulse hir tensorflow onnx-opl onnx kaldi libcli cli ffi"

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

# set_version cargo-dinghy/Cargo.toml 0.3.0
set_version() {
    FILE=$1
    VERSION=$2
    toml set $FILE "package.version" $VERSION > $FILE.tmp
    mv $FILE.tmp $FILE
    for dep in $CRATES
    do
        if toml get $FILE dependencies.tract-$dep
        then
            toml set $FILE "dependencies.tract-$short_dep.version" $VERSION > $FILE.tmp
            mv $FILE.tmp $FILE
        fi
    done
}

set -ex

for c in $CRATES
do
    set_version $c/Cargo.toml $VERSION
done

git commit . -m "post-release $VERSION"
