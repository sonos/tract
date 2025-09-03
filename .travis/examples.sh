#!/bin/sh

WHITE='\033[1;37m'
NC='\033[0m' # No Color

set -e

ROOT=$(dirname $(dirname $(realpath $0)))
. $ROOT/.travis/ci-system-setup.sh

for t in `find examples -name ci.sh`
do
    df -h
    ex=$(dirname $t)
    echo ::group:: $ex
    echo $WHITE $ex $NC
    ( cd $ex ; sh ./ci.sh )
    if [ -n "$CI" ]
    then
        cargo clean
    fi
    echo ::endgroup::
done

