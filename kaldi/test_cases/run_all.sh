#!/bin/bash

set -e

TEST_CASES=$(dirname $0)
FAILURES=""
FAILED=()

cd $TEST_CASES
for tc in */
do
    echo -n "$tc "
    . $tc/vars.sh
    cmd="cargo run -q -p tract -- \
        -f kaldi $tc/model.raw.txt \
        --input-bundle $tc/io.npz \
        --kaldi-downsample $subsampling \
        --kaldi-left-context $left_context \
        --kaldi-right-context $right_context \
        run \
        --assert-output-bundle $tc/io.npz"

    if $($cmd 2> /dev/null > /dev/null)
    then
        echo -e "\e[92mOK\e[39m"
    else
        echo -e "\e[91mFAIL\e[39m"
        FAILED+=("$cmd")
        FAILURES="$FAILURES $tc"
    fi
done

if [ -n "$FAILURES" ]
then
    echo 
    echo -e "    \e[91m$(echo $FAILURES | wc -w) FAILURES\e[39m"
    echo
fi

for cmd in "${FAILED[@]}"
do
    echo $cmd
done

[ -n "$FAILURES" ] || exit 1
