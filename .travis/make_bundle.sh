#!/bin/sh

set -ex

TRAVIS_COMMIT=${TRAVIS_COMMIT:-dummy-commit-id}
TRAVIS_BRANCH=${TRAVIS_BRANCH:-dummy-branch}
PLATFORM=${PLATFORM:-dummy-platform}

dates=`date -u +"%Y%m%dT%H%M%S %s"`
date_iso=`echo $dates | cut -f 1 -d ' '`
timestamp=`echo $dates | cut -f 2 -d ' '`
TASK_NAME=tract-$date_iso
mkdir -p $TASK_NAME
echo "export TASK_NAME=$TASK_NAME" > $TASK_NAME/vars
echo "export TRAVIS_COMMIT=$TRAVIS_COMMIT" >> $TASK_NAME/vars
TRAVIS_BRANCH_SANE=`echo $TRAVIS_BRANCH | tr '/' '_'`
echo "export TRAVIS_BRANCH_SANE=$TRAVIS_BRANCH_SANE" >> $TASK_NAME/vars
echo "export DATE_ISO=$date_iso" >> $TASK_NAME/vars
echo "export TIMESTAMP=$timestamp" >> $TASK_NAME/vars
echo "export PLATFORM=$PLATFORM" >> $TASK_NAME/vars

mkdir $TASK_NAME/benches
cp target/$RUSTC_TRIPLE/release/tract $TASK_NAME
cp target/$RUSTC_TRIPLE/release/mm_for_wavenet_hw-???????????????? $TASK_NAME/benches/mm_for_wavenet_hw
cp .travis/bundle-entrypoint.sh $TASK_NAME/entrypoint.sh
tar czf $TASK_NAME.tgz $TASK_NAME/

echo $TASK_NAME
