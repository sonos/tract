#!/bin/sh

CRATE=$1
VERSION=$2
CRATES="linalg core nnef pulse-opl pulse hir tensorflow onnx-opl onnx kaldi cli"

if [ `uname` = "Darwin" ]
then
    SED=gsed
else
    SED=sed
fi

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
    $SED -i.back "0,/^version/s/^version *= *\".*\"/version = \"$2\"/" $FILE
    for dep in `grep "^tract-" $FILE | cut -d " " -f 1`
    do
        cargo add --manifest-path $FILE $dep@$VERSION
    done
}

for crate in $CRATES
do
    cargo update -p tract-$crate || /bin/true
done

set_version $CRATE/Cargo.toml $VERSION
(cd $CRATE ; cargo publish --allow-dirty)

if [ "$CRATE" = "cli" ]
then
    git commit -m "release $VERSION" .
    git tag -f v"$VERSION"
    git push -f --tags
fi

cargo update
sleep 10
