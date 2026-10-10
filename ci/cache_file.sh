#!/bin/sh

set -e

: ${TRACT_MODELS_URL:=https://tract-test-assets.tract.rs}

if [ -z "$CACHEDIR" ]
then
    CACHEDIR=$HOME/.cache/tract-test-assets
fi

mkdir -p $CACHEDIR
cd $CACHEDIR
for file in $@
do
    mkdir -p $(dirname $file)
    if [ ! -e $file ]
    then
        wget --no-verbose "$TRACT_MODELS_URL/$file" -O $file.tmp
        mv $file.tmp $file
    fi
done

exit 0
