#!/bin/sh

set -ex

cargo install cbindgen

cbindgen ffi > tract.h
cp tract.h c
