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

: ${TRACT_RUN:=cargo run -p tract $CARGO_OPTS --}

for tc in $TEST_CASES
do
    if [ ! -e "$tc/vars.sh" ]
    then
        continue
    fi
    . $tc/vars.sh
    for pass in decl opti
    do
        [[ "$pass" = "opti" ]] && opti="-O" || opti=""
        echo -n "$tc ($pass) "
        cmd="$TRACT_RUN \
            $tc/model.onnx$suffix \
            --output-node output \
            --input-bundle $tc/io.npz \
            --kaldi-downsample 1 \
            --kaldi-left-context $left_context \
            --kaldi-right-context $right_context \
            --kaldi-adjust-final-offset $adjust_final_offset \
            $opti \
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
