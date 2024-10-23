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
    [ -e $file ] \
        || wget --no-verbose https://s3.amazonaws.com/tract-ci-builds/tests/$file -O $file \
        || aws s3 cp s3://tract-ci-builds/tests/$file $file
done

exit 0
