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
    mkdir -p $(dirname $file)
    if [ ! -e $file ]
    then
        wget --no-verbose https://s3.amazonaws.com/tract-ci-builds/tests/$file -O $file.tmp \
        || aws s3 cp s3://tract-ci-builds/tests/$file $file.tmp
        mv $file.tmp $file
    fi
done

exit 0
