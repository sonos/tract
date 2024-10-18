#!/bin/bash

set -e

ROOT=$(dirname $(dirname $(realpath $0)))
. $ROOT/.travis/ci-system-setup.sh

if [ -z $TRACT_RUN ]
then
    TRACT_RUN=$(cargo build --message-format json -p tract $CARGO_EXTRA --profile opt-no-lto --no-default-features | jq -r 'select(.target.name == "tract" and .executable).executable')
    export TRACT_RUN
fi

echo TRACT_RUN=$TRACT_RUN
model=$1
q=$2
generation=current

case $model in
    OpenELM-270M) id=apple--OpenELM-270M-$q;;
    OpenELM-1_1B) id=apple--OpenELM-1_1B-$q;;
    TinyLlama_v1.1) id=TinyLlama--TinyLlama_v1.1-$q;;
    phi-1_5) id=microsoft--phi-1_5-$q;;
    *)
        echo "Unknown model"
        exit 2
        ;;
esac

nnef=llm/$generation/$id/$id.nnef.tgz
pp=llm/$generation/$id/$id.pp.io.npz
tg=llm/$generation/$id/$id.tg.io.npz

set -x
$CACHE_FILE $nnef $pp $tg

$TRACT_RUN -v --nnef-tract-core $MODELS/$nnef -O run \
    --input-from-npz $MODELS/$pp \
    --assert-output-bundle $MODELS/$pp \
    --approx very --allow-float-casts

$TRACT_RUN -v --nnef-tract-core $MODELS/$nnef -O run \
    --input-from-npz $MODELS/$tg \
    --assert-output-bundle $MODELS/$tg \
    --approx very --allow-float-casts
