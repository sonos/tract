#!/bin/sh

set -ex

`dirname $0`/native.sh
cd `dirname $0`/../examples
for i in *
do
    (cd $i; cargo test --release)
done
