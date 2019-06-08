# vim: set syntax=Dockerfile:

FROM tensorflow/tensorflow:devel

RUN apt-get update ; apt-get upgrade -y
RUN apt-get install -y crossbuild-essential-arm64
COPY linux_makefile.inc /tensorflow_src/tensorflow/lite/tools/make/targets/linux_makefile.inc
COPY disable_nnapi.patch /tensorflow_src

WORKDIR /tensorflow_src
RUN patch -p1 < disable_nnapi.patch
