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
    if [ -n $AWS_ACCESS_KEY_ID ]
    then
        [ -e $file ] || aws s3 cp s3://tract-ci-builds/tests/$file $file
    else
        [ -e $file ] || wget https://s3.amazonaws.com/tract-ci-builds/tests/$file -O $file
    fi
done

exit 0
