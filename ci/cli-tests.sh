#!/bin/sh

WHITE='\033[1;37m'
NC='\033[0m' # No Color

set -e

ROOT=$(dirname $(dirname $(realpath $0)))
. $ROOT/ci/ci-system-setup.sh

TRACT_RUN=$(cargo build --message-format json -p tract-cli $CARGO_EXTRA --profile opt-no-lto | jq -r 'select(.target.name == "tract" and .executable).executable')
echo TRACT_RUN=$TRACT_RUN
export TRACT_RUN

for t in $(find $ROOT/harness -name ci.sh | sort)
do
    echo $WHITE$t$NC
    $t
done
