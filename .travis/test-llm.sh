#!/bin/bash

set -e
set -o pipefail

export LC_ALL=C

ROOT=$(dirname $(dirname $(realpath $0)))
. $ROOT/.travis/ci-system-setup.sh

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

id=$model_id-$q

if which gstat > /dev/null
then
    STAT=gstat
else
    STAT=stat
fi

set -x

nnef=llm/$generation/$id/$id.nnef.tgz

$CACHE_FILE $nnef

if [ -e $MODELS/$nnef ]
then
    size=$($STAT -c %s $MODELS/$nnef)
else
    size=$(curl -s -I $MODELS/$nnef | grep Content-Length | cut -d " " -f 2 | tr -cd 0123456789)
fi

if which nvidia-smi > /dev/null
then
    vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{print $1*1024*1024}')
    if [ $vram -lt $size ]
    then
        echo "::warning::Skipping this test, not enough VRAM."
        exit 0
    fi
fi

$TRACT_RUN -v --nnef-tract-transformers $MODELS/$nnef -O --readings  --assert-maximal-mm-quality-cost 0 $TRACT_EXTRA_ARGS dump -q
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
        $TRACT_RUN -v $MODELS/$nnef $TRACT_EXTRA_ARGS \
            --llm --transform unfold-kv-cache -O $DEVICE run --prompt-chunk-size 60 --allow-missing-outputs \
            --input-from-npz $MODELS/$npz \
            --assert-output-bundle $MODELS/$npz \
            --assert-llm-rbo 0.0 \
            $approx --allow-float-casts 2>&1 | tee output.txt
        found=$(cat output.txt | perl -ne 'printf("%.2f\n", $1) if /LLM RBO:\s+([\d.]+)/')
        ( ( grep -v $key $expectations || true) ; echo $key $found) | sort > $expectations.tmp
        mv $expectations.tmp $expectations
    elif [ -n "$RELAX" ]
    then
        prior=$(grep $key $expectations | cut -f 2 -d ' ')
        $TRACT_RUN -v $MODELS/$nnef $TRACT_EXTRA_ARGS \
            --llm --transform unfold-kv-cache -O $DEVICE run --prompt-chunk-size 60 --allow-missing-outputs \
            --input-from-npz $MODELS/$npz \
            --assert-output-bundle $MODELS/$npz \
            --assert-llm-rbo 0.0 \
            $approx --allow-float-casts 2>&1 | tee output.txt
        found=$(cat output.txt | perl -ne 'printf("%.2f\n", $1) if /LLM RBO:\s+([\d.]+)/')
        if [ -n "$prior" ] && perl -e 'exit($ARGV[0] <= $ARGV[1] ? 1 : 0)' "$found" "$prior"
        then
            found=$prior
        fi
        ( ( grep -v $key $expectations || true) ; echo $key $found) | sort > $expectations.tmp
        mv $expectations.tmp $expectations
    else # test !
        expectation=$(grep $key $expectations | cut -f 2 -d ' ')
        $TRACT_RUN -v $MODELS/$nnef $TRACT_EXTRA_ARGS \
            --llm --transform unfold-kv-cache -O $DEVICE run --prompt-chunk-size 60 --allow-missing-outputs \
            --input-from-npz $MODELS/$npz \
            --assert-output-bundle $MODELS/$npz \
            --assert-llm-rbo $expectation \
            $approx --allow-float-casts
    fi

done
