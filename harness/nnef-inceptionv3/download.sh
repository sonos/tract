#!/bin/sh

MY_DIR=`dirname $0`

$MY_DIR/../../.travis/cache_file.sh inception_v3.tfpb.nnef.tgz imagenet_slim_labels.txt
