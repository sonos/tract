#!/bin/sh

set -ex

cbindgen ffi > tract,h
cp tract.h c
cp tract.h proxy/sys
