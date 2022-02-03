#!/bin/sh

set -ex

TRAVIS_COMMIT=${GITHUB_SHA:-dummy-commit-id}
BRANCH=$(echo $GITHUB_HEAD_REF | tr '/' '_')
BRANCH=${BRANCH:-main}
PLATFORM=${PLATFORM:-dummy-platform}

dates=`date -u +"%Y%m%dT%H%M%S %s"`
date_iso=`echo $dates | cut -f 1 -d ' '`
timestamp=`echo $dates | cut -f 2 -d ' '`
TASK_NAME=tract-$date_iso
mkdir -p $TASK_NAME
echo "export TASK_NAME=$TASK_NAME" > $TASK_NAME/vars
echo "export TRAVIS_COMMIT=$TRAVIS_COMMIT" >> $TASK_NAME/vars
TRAVIS_BRANCH_SANE=`echo $BRANCH | tr '/' '_'`
echo "export TRAVIS_BRANCH_SANE=$TRAVIS_BRANCH_SANE" >> $TASK_NAME/vars
echo "export DATE_ISO=$date_iso" >> $TASK_NAME/vars
echo "export TIMESTAMP=$timestamp" >> $TASK_NAME/vars
echo "export PLATFORM=$PLATFORM" >> $TASK_NAME/vars

touch sizes
for bin in example-tensorflow-mobilenet-v2 tract
do
    if [ -e target/$RUSTC_TRIPLE/release/$bin ]
    then
        binary_size_cli=$(stat -c "%s" target/$RUSTC_TRIPLE/release/$bin)
        token=$(echo $bin | tr '-' '_')
        if [ "$bin" = "tract" ]
        then
            token=cli
        fi
        echo binary_size.$token $binary_size_cli >> sizes
    fi
done

cp target/$RUSTC_TRIPLE/release/tract $TASK_NAME
cp sizes $TASK_NAME
cp .travis/bundle-entrypoint.sh $TASK_NAME/entrypoint.sh
tar czf $TASK_NAME.tgz $TASK_NAME/

echo $TASK_NAME
