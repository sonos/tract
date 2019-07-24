#!/bin/sh

set -ex

if [ -z "$PLATFORM" ]
then
  ts -s .travis/native.sh
else
  .travis/cross.sh
fi
