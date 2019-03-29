#!/bin/sh

MY_DIR=`dirname $0`

$MY_DIR/../../.travis/cache_file.sh deepspeech-0.4.1.pb deepspeech-0.4.1-smoketest.txt
