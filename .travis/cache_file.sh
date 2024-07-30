#!/bin/sh

set -e

if [ -z "$CACHEDIR" ]
then
    CACHEDIR=`dirname $0`/../.cached
fi

mkdir -p $CACHEDIR

cd $CACHEDIR
for file in $@
do
     [ -e $file ] || wget https://s3.amazonaws.com/tract-ci-builds/tests/$file
done

exit 0
