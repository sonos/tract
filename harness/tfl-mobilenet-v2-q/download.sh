#!/bin/sh

MY_DIR=`dirname $0`

$MY_DIR/../../.travis/cache_file.sh \
    mobilenetv2_ptq_single_img.tflite \
    imagenet_slim_labels.txt \
    grace_hopper.jpg
