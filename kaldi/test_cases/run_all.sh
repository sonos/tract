#!/bin/bash

set -e

TEST_CASE_DIR=$(dirname $0)
FAILURES=""
FAILED=()

if [ "$#" -gt 0 ]
then
    TEST_CASES="$@"
else
    TEST_CASES="$TEST_CASE_DIR/*"
fi

for tc in $TEST_CASES
do
    if [ ! -e "$tc/vars.sh" ]
    then
        continue
    fi
    . $tc/vars.sh
    for form in txt bin
    do
        [[ "$form" = "txt" ]] && suffix=.txt || suffix=""
        echo -n "$tc ($form) "
        cmd="cargo run -q -p tract $CARGO_OPTS -- \
            -f kaldi $tc/model.raw$suffix \
            --output-node output \
            --input-bundle $tc/io.npz \
            --kaldi-downsample $subsampling \
            --kaldi-left-context $left_context \
            --kaldi-right-context $right_context \
            --kaldi-adjust-final-offset $adjust_final_offset \
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

[ -z "$FAILURES" ]
