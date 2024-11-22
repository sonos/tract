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

nnef=llm/$generation/$id/$id.nnef.tgz

$CACHE_FILE $nnef

$TRACT_RUN -v --nnef-tract-core $MODELS/$nnef -O --readings dump -q
rszmax=$(tail -1 readings.out | awk '{print $5}')
limit=$(zcat $MODELS/$nnef | wc -c)
ratio=$((rszmax * 100 / limit))

if [ $ratio -gt 175 ]
then
    echo "RSZ max is ${ratio}% the size of the unzipped model!"
    exit 1
fi

set -x

for t in p0s100 p50s50 p99s1 
do
    npz=llm/$generation/$id/$id.$t.io.npz
    $CACHE_FILE $npz

    case $q in
        q40f16) approx="--approx ultra";;
        q40ef16) approx="--approx ultra";;
        f16f16) approx="--approx ultra";;
        q40f32) approx="--approx very";;
        q40ef32) approx="--approx very";;
        f32f32) approx="--approx approximate";;
    esac

    case "$id.$t" in 
        apple--OpenELM-270M-f16f16.p50s50) approx="--approx-custom 0.2,0.1,0.003";;
        TinyLlama--TinyLlama_v1.1-f16f16.p0s100) approx="--approx-custom 0.2,0.1,0.001";;
        TinyLlama--TinyLlama_v1.1-f16f16.p50s50) approx="--approx-custom 0.2,0.1,0.005";;
        TinyLlama--TinyLlama_v1.1-f16f16.p99s1) approx="--approx-custom 0.2,0.1,0.004";;
    esac


    $TRACT_RUN -v --nnef-tract-core $MODELS/$nnef -O run \
        --input-from-npz $MODELS/$npz \
        --assert-output-bundle $MODELS/$npz \
        $approx --allow-float-casts
done
