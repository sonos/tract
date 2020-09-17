#!/bin/sh

VERSION=$1
CRATES="linalg core nnef pulse-opl pulse hir tensorflow onnx-opl onnx kaldi cli"

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
    $SED -i.back "0,/^version/s/^version *= *\".*\"/version = \"$2\"/" $FILE
    for dep in `grep "^tract-" $FILE | cut -d " " -f 1`
    do
        short_dep=`echo $dep | sed "s/^tract-//"`
        cargo add --manifest-path $FILE --path ../$short_dep tract-$short_dep
    done
}

set -ex

for c in $CRATES
do
    set_version $c/Cargo.toml $VERSION
done

git commit . -m "post-release $VERSION"
