#!/bin/sh

set -ex

cargo install cbindgen --version 0.29.4 --locked

cbindgen ffi > tract.h
cp tract.h c
mv tract.h proxy/sys
