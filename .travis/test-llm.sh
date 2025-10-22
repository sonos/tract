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
generation=541

if [ "$model" = "all" ]
then
    for m in \
        openelm-270M \
	llama-3.2-1B-instruct \
	llama-3.2-3B-instruct \
	llama-3.1-8B-instruct \
	qwen2.5-7B-instruct \
	qwen3-1.7B \
	qwen3-8B
    do
        $0 $m $2 $device
    done
    exit 0
fi

model=$(echo $model | tr 'A-Z' 'a-z' | tr -d "_.-")

for m in \
    apple--OpenELM-270M \
    meta-llama--Llama-3.2-1B-Instruct \
    meta-llama--Llama-3.2-3B-Instruct \
    meta-llama--Llama-3.1-8B-Instruct \
    Qwen--Qwen2.5-7B-Instruct \
    Qwen--Qwen3-1.7B \
    Qwen--Qwen3-8B
do
    norm=$(echo $m | tr "A-Z" "a-z" | tr -d "_.-")
    if [[ "$norm" == *"$model"* ]];
    then
        model_id=$m
    fi
done

if [ -z "$model_id" ]
then
    echo "No model matched"
fi

if [ "$q" = "all" ]
then
    for q in q40ef16 f16f16 f32f32
    do
        $0 $1 $q $device
    done
    exit 0
fi

# Skipping f32f32 models except LLama 1B
MODELS_F32_ALLOWED="llama-3.2-1B-instruct"
if [ "$q" = "f32f32" ] && ! echo "$MODELS_F32_ALLOWED" | grep -q -w "$1"
then
    echo "INFO: Skipping f32f32 for model $1."
    exit 0
fi

id=$model_id-$q

# Skipping too big models for CI workers
TOO_BIG_MODELS=(
    "meta-llama--Llama-3.1-8B-Instruct-f32f32:cuda"
    "meta-llama--Llama-3.1-8B-Instruct-f16f16:cuda"
    "Qwen--Qwen2.5-7B-Instruct-f32f32:cuda"
    "Qwen--Qwen2.5-7B-Instruct-f16f16:cuda"
    "Qwen--Qwen3-8B-f32f32:cuda"
    "Qwen--Qwen3-8B-f16f16:cuda"
)

for big_id in "${TOO_BIG_MODELS[@]}"
do
    if [ "$big_id" = "$id:$device" ]
    then
        echo "INFO: Skipping model $id because it is too big for $device CI worker."
        exit 0
    fi
done

if [ -n "$GITHUB_ACTIONS" ]
then
    for m in \
        meta-llama--Llama-3.1-8B-Instruct-f32f32 \
        meta-llama--Llama-3.1-8B-Instruct-f16f16 \
        Qwen--Qwen2.5-7B-Instruct-f32f32 \
        Qwen--Qwen2.5-7B-Instruct-f16f16 \
        Qwen--Qwen3-8B-f32f32 \
        Qwen--Qwen3-8B-f16f16
    do
        if [[ "$m" = "$id" ]]
	then
            echo "::warning title=Untestable model::$id is too big for GHA..."
            exit 0
	fi
    done
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

$TRACT_RUN -v --nnef-tract-transformers $MODELS/$nnef -O --readings  --assert-maximal-mm-quality-cost 0 $TRACT_EXTRA_ARGS dump -q
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
    expectations="$ROOT/.travis/llm-expectations-541"

    case $device in 
        cuda)
            DEVICE="--cuda"
        ;;
        metal) DEVICE="--metal";;
    esac

    if [ -n "$RESET" ]
    then
        $TRACT_RUN -v --nnef-tract-core --nnef-tract-transformers $MODELS/$nnef $TRACT_EXTRA_ARGS \
            -t transformers-detect-all -O $DEVICE run \
            --input-from-npz $MODELS/$npz \
            --assert-output-bundle $MODELS/$npz \
            --assert-llm-lev20 999999999 \
            $approx --allow-float-casts 2>&1 | tee output.txt
        found=$(cat output.txt | grep lev20 | cut -d '=' -f 2)
        ( ( grep -v $key $expectations || true) ; echo $key $found) | sort > $expectations.tmp
        mv $expectations.tmp $expectations
    elif [ -n "$RELAX" ]
    then
        prior=$(grep $key $expectations | cut -f 2 -d ' ')
        $TRACT_RUN -v --nnef-tract-core --nnef-tract-transformers $MODELS/$nnef $TRACT_EXTRA_ARGS \
            -t transformers-detect-all -O $DEVICE run \
            --input-from-npz $MODELS/$npz \
            --assert-output-bundle $MODELS/$npz \
            --assert-llm-lev20 999999999 \
            $approx --allow-float-casts 2>&1 | tee output.txt
        found=$(cat output.txt | grep lev20 | cut -d '=' -f 2)
        if [ "$found" -lt "$prior" ]
        then
            found=$prior
        fi
        ( ( grep -v $key $expectations || true) ; echo $key $found) | sort > $expectations.tmp
        mv $expectations.tmp $expectations
    else # test !
        expectation=$(grep $key $expectations | cut -f 2 -d ' ')
        $TRACT_RUN -v --nnef-tract-core --nnef-tract-transformers $MODELS/$nnef $TRACT_EXTRA_ARGS \
            -t transformers-detect-all -O $DEVICE run \
            --input-from-npz $MODELS/$npz \
            --assert-output-bundle $MODELS/$npz \
            --assert-llm-lev20 $expectation \
            $approx --allow-float-casts
    fi

done
