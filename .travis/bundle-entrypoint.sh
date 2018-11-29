#!/bin/sh

set -ex

ROOT=`pwd`
CACHEDIR=${CACHEDIR:-$HOME/.cache}
mkdir -p $CACHEDIR

echo "Hey! I'm the entrypoint!"
./tract --help

. ./vars

(
    cd $CACHEDIR
    if [ ! -e inception_v3_2016_08_28_frozen.pb.tar.gz ]
    then
        wget http://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz
    fi
    if [ !  -e inception_v3_2016_08_28_frozen.pb ]
    then
        tar zxf inception_v3_2016_08_28_frozen.pb.tar.gz
    fi
)

./tract $CACHEDIR/inception_v3_2016_08_28_frozen.pb -O -i 1x299x299x3xf32 profile --bench

tract_size=`stat -c "%s" tract`

echo binary_size.cli.$PLATFORM.$TRAVIS_BRANCH_SANE $tract_size $TIMESTAMP > metrics
