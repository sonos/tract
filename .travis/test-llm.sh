#!/bin/bash

set -e
set -o pipefail

ROOT=$(dirname $(dirname $(realpath $0)))
. $ROOT/.travis/ci-system-setup.sh

if [ -z "$TRACT_RUN" ]
then
    TRACT_RUN=$(cargo build --message-format json -p tract $CARGO_EXTRA --profile opt-no-lto --no-default-features --features transformers | jq -r 'select(.target.name == "tract" and .executable).executable')
    export TRACT_RUN
fi

echo TRACT_RUN=$TRACT_RUN
model=$1
q=$2
device=$3
generation=516

case $model in
    OpenELM-270M) id=apple--OpenELM-270M-$q;;
    OpenELM-1_1B) id=apple--OpenELM-1_1B-$q;;
    TinyLlama_v1.1) id=TinyLlama--TinyLlama_v1.1-$q;;
    llama-3.2-3B) id=meta-llama--Llama-3.2-3B-$q;;
    llama-3.2-1B) id=meta-llama--Llama-3.2-1B-$q;;
    *)
        echo "Unknown model"
        exit 2
        ;;
esac

if [ -n "$GITHUB_ACTIONS" ]
then
    if [ "$id" =  meta-llama--Llama-3.2-3B-f32f32 ]
    then
        echo "::warning title=Untestable model::$id is too big for GHA..."
        exit 0
    fi
fi

if which gstat > /dev/null
then
    STAT=gstat
else
    STAT=stat
fi

set -x

nnef=llm/$generation/$id/$id.nnef.tgz

$CACHE_FILE $nnef

$TRACT_RUN -v --nnef-tract-core $MODELS/$nnef -O --readings  --assert-maximal-mm-quality-cost 0 $TRACT_EXTRA_ARGS dump -q
if [ -e $MODELS/$nnef ]
then
    size=$($STAT -c %s $MODELS/$nnef)
else
    size=$(curl -s -I $MODELS/$nnef | grep Content-Length | cut -d " " -f 2 | tr -cd 0123456789)
fi

alloc_max=$(cat readings.out | tail -n +2 | awk '{print $10-$11}' | sort -n | tail -1)
ratio=$((alloc_max * 100 / size))

echo "  ###########################################"
echo "      Alloc max to model size ratio: ${ratio}%."
echo "  ###########################################"

limit=125

if [ $ratio -gt $limit ]
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
        f32f32) approx="--approx very";;
    esac

    case "$id.$t" in
        # very terrible conditioning on this one
        apple--OpenELM-270M-f16f16.p50s50) approx="--approx-custom 0.2,0.25,0.013";;

        TinyLlama--TinyLlama_v1.1-f32f32.p50s50) approx="--approx-custom 0.2,0.1,0.001";;
        TinyLlama--TinyLlama_v1.1-f16f16.p0s100) approx="--approx-custom 0.2,0.1,0.002";;
        TinyLlama--TinyLlama_v1.1-f16f16.p50s50) approx="--approx-custom 0.2,0.1,0.007";;
        TinyLlama--TinyLlama_v1.1-f16f16.p99s1) approx="--approx-custom 0.2,0.1,0.004";;
        TinyLlama--TinyLlama_v1.1-q40f16.p0s100) approx="--approx-custom 0.2,0.1,0.004";;
        TinyLlama--TinyLlama_v1.1-q40f16.p99s1) approx="--approx-custom 0.2,0.1,0.002";;
        TinyLlama--TinyLlama_v1.1-q40f16.p50s50) approx="--approx-custom 0.2,0.2,0.006";;
        TinyLlama--TinyLlama_v1.1-q40ef16.p0s100) approx="--approx-custom 0.2,0.1,0.002";;
        TinyLlama--TinyLlama_v1.1-q40ef16.p50s50) approx="--approx-custom 0.2,0.1,0.002";;

        meta-llama--Llama-3.2-3B-f16f16.p0s100 |\
        meta-llama--Llama-3.2-3B-q40f16.p0s100 |\
        meta-llama--Llama-3.2-3B-q40ef16.p0s100) 
            if [ `arch` = "arm64" ]
            then
                approx="--approx-custom 0.25,0.25,0.01"
            else
                approx="--approx-custom 0.2,0.1,0.004"
            fi
        ;;
        meta-llama--Llama-3.2-3B-f16f16.p50s50 |\
        meta-llama--Llama-3.2-3B-q40f16.p50s50 |\
        meta-llama--Llama-3.2-3B-q40ef16.p50s50) 
            if [ `arch` = "arm64" ]
            then
                approx="--approx-custom 0.25,0.25,0.016"
            else
                approx="--approx-custom 0.2,0.1,0.004"
            fi
        ;;
    esac

    case $device in 
        cuda) case "$id.$t" in
                TinyLlama--TinyLlama_v1.1-q40f16.p99s1) approx="--approx-custom 0.25,0.25,0.006";;

                TinyLlama--TinyLlama_v1.1-q40f32.p0s100) approx="--approx-custom 0.5,0.5,0.015";;
                TinyLlama--TinyLlama_v1.1-q40f32.p50s50) approx="--approx-custom 0.2,0.1,0.001";;
                TinyLlama--TinyLlama_v1.1-q40f32.p99s1) approx="--approx-custom 0.35,0.25,0.007";;

                TinyLlama--TinyLlama_v1.1-q40ef16.p99s1) approx="--approx-custom 0.35,0.25,0.005";;

                TinyLlama--TinyLlama_v1.1-q40ef32.p0s100) approx="--approx ultra";;
                TinyLlama--TinyLlama_v1.1-q40ef32.p50s50) approx="--approx-custom 0.2,0.1,0.003";;
                TinyLlama--TinyLlama_v1.1-q40ef32.p99s1) approx="--approx-custom 0.35,0.25,0.012";;

                meta-llama--Llama-3.2-3B-q40f32.p0s100 |\
                meta-llama--Llama-3.2-3B-q40f32.p50s50) approx="--approx ultra";;
                meta-llama--Llama-3.2-3B-q40f32.p99s1) approx="--approx super";;

                meta-llama--Llama-3.2-3B-q40ef32.p0s100) approx="--approx-custom 0.25,0.15,0.001";;
                meta-llama--Llama-3.2-3B-q40ef32.p50s50 |\
                meta-llama--Llama-3.2-3B-q40ef32.p99s1) approx="--approx ultra";;

                apple--OpenELM-1_1B-q40f16.p99s1) approx="--approx-custom 0.25,0.1,0.001";;

                apple--OpenELM-1_1B-q40f32.p0s100) approx="--approx super";;
                apple--OpenELM-1_1B-q40f32.p50s50) approx="--approx-custom 0.25,0.1,0.001";;
                apple--OpenELM-1_1B-q40f32.p99s1) approx="--approx-custom 0.25,0.25,0.0025";;

                apple--OpenELM-1_1B-q40ef16.p99s1) approx="--approx-custom 0.25,0.2,0.002";;

                apple--OpenELM-1_1B-q40ef32.p0s100) approx="--approx super";;
                apple--OpenELM-1_1B-q40ef32.p50s50) approx="--approx-custom 0.2,0.1,0.001";;
                apple--OpenELM-1_1B-q40ef32.p99s1) approx="--approx-custom 0.25,0.2,0.002";;

                apple--OpenELM-270M-q40f16.p99s1 ) approx="--approx-custom 0.25,0.15,0.002";;

                apple--OpenELM-270M-q40f32.p0s100 |\
                apple--OpenELM-270M-q40f32.p50s50) approx="--approx super";;
                apple--OpenELM-270M-q40f32.p99s1) approx="--approx-custom 0.25,0.15,0.001";;

                ##Fix this
                ##apple--OpenELM-270M-q40ef16.p0s100 ) approx="--approx-custom 0.5,0.5,0.015";;
                ##apple--OpenELM-270M-q40ef16.p50s50 ) approx="--approx ultra";;
                ##apple--OpenELM-270M-q40ef16.p99s1) approx="--approx ultra";;
##
                ##apple--OpenELM-270M-q40ef32.p0s100 ) approx="--approx-custom 0.25,0.15,0.001";;
                ##apple--OpenELM-270M-q40ef32.p50s50 ) approx="--approx-custom 0.2,0.1,0.001";;
                ##apple--OpenELM-270M-q40ef32.p99s1) approx="--approx-custom 0.2,0.1,0.001";;

              esac
              DEVICE="--cuda";;
        metal) DEVICE="--metal";;
    esac

    $TRACT_RUN -v --nnef-tract-core --nnef-tract-transformers $MODELS/$nnef $TRACT_EXTRA_ARGS \
        -t transformers-detect-all -O $DEVICE run \
        --input-from-npz $MODELS/$npz \
        --assert-output-bundle $MODELS/$npz \
        $approx --allow-float-casts
done
