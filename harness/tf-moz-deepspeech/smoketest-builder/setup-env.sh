#!/bin/sh

if [ ! -e DeepSpeech ]
then 
    git clone https://github.com/mozilla/DeepSpeech.git
fi

if [ ! -e tensorflow ]
then 
    git clone https://github.com/mozilla/tensorflow.git
    (cd tensorflow ;
    git checkout r1.13 ;
    ln -s ../DeepSpeech/native_client .
    )
fi

docker build --tag deepspeech .
