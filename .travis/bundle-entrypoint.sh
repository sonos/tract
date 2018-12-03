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

binary_size_cli=`stat -c "%s" tract`
inceptionv3=`./tract $CACHEDIR/inception_v3_2016_08_28_frozen.pb -O -i 1x299x299x3xf32 profile --bench | grep Real | perl -pe 's/\x1b\[[0-9;]*m//g' | cut -f 2 -d " "`


echo binary_size.cli $binary_size_cli > metrics
echo net.inceptionv3.evaltime.full $inceptionv3 > metrics
