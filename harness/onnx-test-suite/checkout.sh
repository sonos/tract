#!/bin/sh

set -ex

if [ -z "$CACHEDIR" ]
then
    CACHEDIR=`dirname $0`/../../.cached
fi

mkdir -p $CACHEDIR/onnx
cd $CACHEDIR/onnx

for version in 1.4.1 1.5.0
do
    if [ ! -e onnx-$version/onnx/backend/test/data ]
    then
        (
            rm -rf onnx-$version
            tmp=$(mktemp -d -p .)
            git clone https://github.com/onnx/onnx $tmp
            (cd $tmp ; git checkout v$version)
            mv $tmp onnx-$version
        )
    fi
done
