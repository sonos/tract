#!/bin/bash

crate=$1
version=$2

perl -pi -e "s/^($crate = {.*version *= *)\"([^\"]*)\"(.*)$/\$1\"=$version\"\$3/" \
    `find . -name Cargo.toml \! -path "./target/*" \! -path "./issue*"`
