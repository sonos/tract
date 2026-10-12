# vim: set syntax=Dockerfile:

FROM tensorflow/tensorflow:nightly-devel

RUN apt-get update ; apt-get upgrade -y
RUN apt-get -y install git crossbuild-essential-armhf

WORKDIR /tensorflow
RUN ./tensorflow/lite/tools/make/download_dependencies.sh


