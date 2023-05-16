#!/bin/bash

set -e

if [ -z "$CACHEDIR" ]
then
    CACHEDIR=`dirname $0`/../../.cached
fi

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
    unset OPTIONS IGNORE MODEL left_context right_context subsampling
    . $tc/vars.sh
    for file in $CACHE_FILES
    do
        $TEST_CASE_DIR/../../.travis/cache_file.sh $file
    done
    : ${MODEL:=$tc/model.onnx}
    for pass in plain decl opti nnef
    do
        printf "$tc ($pass) "
        if [[ $IGNORE == *$pass* ]]
        then
            printf "\e[93mignored\e[39m\n"
            continue
        fi
        case $pass in
            plain) opti="--pass incorporate" ;;
            decl) opti="" ;;
            opti) opti="-O" ;;
            nnef) opti="--nnef-cycle --nnef-tract-core" ;;
        esac
        options="$OPTIONS"
        if [ -n "$left_context" -a "$left_context" != "0" ]
        then
            options="$options --edge-left-context $left_context"
        fi
        if [ -n "$right_context" -a "$right_context" != "0" ]
        then
            options="$options --edge-right-context $right_context"
        fi
        cmd="$TRACT_RUN \
            $MODEL \
            --input-facts-from-bundle $tc/io.npz \
            --onnx-ignore-output-shapes \
            $options \
            $opti \
            run \
            --input-from-bundle $tc/io.npz \
            --assert-output-bundle $tc/io.npz"

        if $($cmd 2> /dev/null > /dev/null)
        then
            printf "\e[92mOK\e[39m\n"
        else
            printf "\e[91mFAIL\e[39m\n"
            FAILED+=("$cmd")
            FAILURES="$FAILURES $tc"
        fi
    done
done

if [ -n "$FAILURES" ]
then
    echo 
    printf "    \e[91m$(echo $FAILURES | wc -w) FAILURES\e[39m\n"
    echo
fi

for cmd in "${FAILED[@]}"
do
    echo $cmd
done

[ -z "$FAILURES" ]
