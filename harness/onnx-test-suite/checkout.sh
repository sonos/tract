#!/bin/sh

set -ex

if [ -z "$CACHEDIR" ]
then
    CACHEDIR=`dirname $0`/../../.cached
fi

mkdir -p $CACHEDIR
cd $CACHEDIR

find .

if [ ! -e onnx/onnx/backend/test/data ]
then
    (
        rm -rf onnx
        uuid=$(uuidgen)
        git clone https://github.com/onnx/onnx onnx-$uuid
        (cd onnx-$uuid ; git checkout v1.4.1)
        mv onnx-$uuid onnx
    )
fi
