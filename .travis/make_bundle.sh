#!/bin/sh

set -ex

if [ -n "$GITHUB_ACTIONS" ]
then
    TRAVIS_COMMIT=${GITHUB_SHA:-dummy-commit-id}
    BRANCH=${TRACT_BENCH_BRANCH_NAME:-${GITHUB_HEAD_REF:-main}}
else
    TRAVIS_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo dummy-commit-id)
    BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)
fi
BRANCH=$(echo $BRANCH | tr '/' '_')
PLATFORM=${PLATFORM:-dummy-platform}
TARGET_DIR="${CARGO_TARGET_DIR:-target}"

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

cp $TARGET_DIR/$RUSTC_TRIPLE/release/tract $TASK_NAME
cp .travis/bundle-entrypoint.sh $TASK_NAME/entrypoint.sh
tar czf $TASK_NAME.tgz $TASK_NAME/

echo $TASK_NAME
