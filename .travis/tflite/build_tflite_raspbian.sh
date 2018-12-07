#!/bin/sh

set -ex
mkdir -p result

# build pseudo-rpi official tensorflow, https://www.tensorflow.org/lite/rpi, only works on pi3

docker build -f Dockerfile.tensorflow-official-rpi --tag tensorflow-official-rpi .
docker run --rm \
    -e CC_PREFIX=arm-linux-gnueabihf- \
    -v `pwd`/result:/result \
    tensorflow-official-rpi \
    sh -c "
        make -j 3 -f tensorflow/lite/tools/make/Makefile TARGET=rpi TARGET_ARCH=armv7l;
        cp /tensorflow/tensorflow/lite/tools/make/gen/rpi_armv7l/bin/benchmark_model /result/tflite_benchmark_model_official_rpi
    "

# build with rpi tools (works on rpi0, 1 and 2)

docker build -f Dockerfile.tensorflow-rpitools --tag tensorflow-rpitools .
docker run --rm \
    -e CC_PREFIX=arm-linux-gnueabihf- \
    -v `pwd`/result:/result \
    tensorflow-rpitools \
    sh -c "
        make -j 3 -f tensorflow/lite/tools/make/Makefile TARGET=rpi TARGET_ARCH=armv6;
        cp /tensorflow/tensorflow/lite/tools/make/gen/rpi_armv6/bin/benchmark_model /result/tflite_benchmark_model_rpitools
    "

aws s3 sync result/ s3://tract-ci-builds/model/
