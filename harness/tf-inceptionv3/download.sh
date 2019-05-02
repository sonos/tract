#!/bin/sh

MY_DIR=`dirname $0`

$MY_DIR/../../.travis/cache_file.sh inception_v3_2016_08_28_frozen.pb imagenet_slim_labels.txt
