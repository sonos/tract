#!/bin/sh

set -ex

MY_DIR=`dirname $0`

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
        git clone https://github.com/onnx/onnx ;
        cd onnx
        git checkout v1.4.1
    )
fi
