#!/bin/sh

set -ex
mkdir -p result

docker build -f Dockerfile.tensorflow-aarch64 --tag tensorflow-aarch64 .
docker run --rm -it \
    -v `pwd`/result:/result \
    tensorflow-aarch64 \
     sh -c "
         cd /tensorflow ;
         make -j 3 -f tensorflow/lite/tools/make/Makefile TARGET=linux TARGET_ARCH=aarch64 ;
         cp /tensorflow/tensorflow/lite/tools/make/gen/linux_aarch64/bin/benchmark_model /result/tflite_benchmark_model_aarch64
     "

aws s3 sync result/ s3://tract-ci-builds/model/
