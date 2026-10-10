#!/bin/sh

set -e

HERE=$(dirname $(realpath $0))
ROOT=$(realpath $HERE/../..)
. $ROOT/.travis/ci-system-setup.sh

for t in $(find $HERE -name runme.sh | sort)
do
    echo $t
    $t
done
