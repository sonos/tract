#!/bin/sh

set -ex

cbindgen ffi > tract,h
cp tract.h c
bindgen tract.h -o proxy/sys/bindings.rs
