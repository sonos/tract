#!/bin/sh

set -ex

if [ -n "$TRACT_TEST" ]
then
  cd examples/$TRACT_TEST
  cargo test --release
else
  if [ -z "$PLATFORM" ]
  then
      .travis/native.sh
  else
      .travis/cross.sh
  fi
fi
