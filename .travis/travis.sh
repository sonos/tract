#!/bin/sh

set -ex

if [ -z "$PLATFORM" ]
then
  .travis/native.sh
else
  .travis/cross.sh
fi
