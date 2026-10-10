#!/bin/sh
# ci-runners: ubuntu-latest cuda-lovelace

set -e

HERE=$(dirname $(realpath $0))
ROOT=$(realpath $HERE/../..)
. $ROOT/.travis/ci-system-setup.sh

# With an accelerator present, only the cases looping over TRACT_RUNTIMES have
# anything to add to the CPU-only run on the hosted runner.
if [ "$TRACT_RUNTIMES" = "-O" ]
then
    cases=$(find $HERE -name runme.sh | sort)
else
    cases=$(grep -rl TRACT_RUNTIMES --include=runme.sh $HERE | sort)
fi

for t in $cases
do
    echo $t
    $t
done
