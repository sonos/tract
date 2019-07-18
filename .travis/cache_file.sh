#!/bin/sh

set -ex

if [ -z "$CACHEDIR" ]
then
    CACHEDIR=`dirname $0`/../.cached
fi

mkdir -p $CACHEDIR

cd $CACHEDIR
for file in $@
do
     [ -e $file ] || wget -q https://s3.amazonaws.com/tract-ci-builds/tests/$file
done
