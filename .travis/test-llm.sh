#!/bin/bash

set -e
set -o pipefail

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

case $q in
    q40f16) approx=ultra;;
    q40ef16) approx=ultra;;
    f16f16) approx=ultra;;
    q40f32) approx=very;;
    q40ef32) approx=very;;
    f32f32) approx=approximate;;
esac

nnef=llm/$generation/$id/$id.nnef.tgz

set -x
$CACHE_FILE $nnef
for t in p0s100 p50s50 p99s1 
do
    npz=llm/$generation/$id/$id.$t.io.npz
    $CACHE_FILE $npz
    $TRACT_RUN -v --nnef-tract-core $MODELS/$nnef -O run \
        --input-from-npz $MODELS/$npz \
        --assert-output-bundle $MODELS/$npz \
        --approx $approx --allow-float-casts
done
