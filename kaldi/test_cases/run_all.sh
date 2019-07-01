#!/bin/sh

set -e

TEST_CASES=$(dirname $0)
FAILURES=""

cd $TEST_CASES
for tc in */
do
    echo -n "$tc "
    if cargo run -q -p tract -- \
        -f kaldi $tc/model.raw.txt \
        --input-bundle $tc/io.npz \
        run \
        --assert-output-bundle $tc/io.npz 2>/dev/null > /dev/null
    then
        echo "\e[92mOK\e[39m"
    else
        echo "\e[91mFAIL\e[39m"
        FAILURES="$FAILURES $tc"
    fi
done

if [ -n "$FAILURES" ]
then
    echo 
    echo "    \e[91m$(echo $FAILURES | wc -w) FAILURES\e[39m"
    echo
fi

for tc in $FAILURES
do
    echo cargo run -p tract -- \
        -f kaldi $tc/model.raw.txt \
        --input-bundle $tc/io.npz \
        run \
        --assert-output-bundle $tc/io.npz
done

[ -n "$FAILURES" ] || exit 1
