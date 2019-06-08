#!/bin/sh

MY_DIR=`dirname $0`

$MY_DIR/../../.travis/cache_file.sh \
    mobilenet_v2_1.4_224_frozen.pb \
    imagenet_slim_labels.txt \
    grace_hopper.jpg
