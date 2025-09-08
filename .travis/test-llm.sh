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
if [ -z "$device" ]
then
    device=cpu
fi
generation=516

case $model in
    all)
        $0 OpenELM-270M $q $device
        $0 OpenELM-1_1B $q $device
        $0 llama-3.2-3B $q $device
        $0 llama-3.2-1B $q $device
        exit 0
    ;;
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

if [ "$q" = "all" ]
then
    for q in q40f16 q40ef16 f16f16 q40f32 q40ef32 f32f32
    do
        $0 $1 $q $device
    done
    exit 0
fi

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

    key=$id.$t.$(arch).$device
    expectations="$ROOT/.travis/llm-expectations-516"

    case $device in 
        cuda) DEVICE="--cuda";;
        metal) DEVICE="--metal";;
    esac

    if [ -z "$RESET" ]
    then
        expectation=$(grep $key $expectations | cut -f 2 -d ' ')
        $TRACT_RUN -v --nnef-tract-core --nnef-tract-transformers $MODELS/$nnef $TRACT_EXTRA_ARGS \
            -t transformers-detect-all -O $DEVICE run \
            --input-from-npz $MODELS/$npz \
            --assert-output-bundle $MODELS/$npz \
            --assert-llm-lev20 $expectation \
            $approx --allow-float-casts
    else
        $TRACT_RUN -v --nnef-tract-core --nnef-tract-transformers $MODELS/$nnef $TRACT_EXTRA_ARGS \
            -t transformers-detect-all -O $DEVICE run \
            --input-from-npz $MODELS/$npz \
            --assert-output-bundle $MODELS/$npz \
            --assert-llm-lev20 999999999 \
            $approx --allow-float-casts 2>&1 | tee output.txt
        found=$(cat output.txt | grep lev20 | cut -d '=' -f 2)
        ( ( grep -v $key $expectations || /bin/true) ; echo $key $found) | sort > $expectations.tmp
        mv $expectations.tmp $expectations
    fi

done
