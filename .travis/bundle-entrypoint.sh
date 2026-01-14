#!/bin/sh

set -ex

start=$(date +%s)

ROOT=`pwd`

if [ -n "$TRACT_RUN" ]
then
    TRACT=$TRACT_RUN
elif [ -x tract ]
then
    TRACT=./tract
else
    cargo build -p tract -q --release
    TRACT=./target/release/tract
fi

CACHEDIR=${CACHEDIR:-$HOME/.cache/tract-ci-minion-models}
case $CACHEDIR in
    "http"*)
        wget $CACHEDIR/private/private-benches.sh
        PRIVATE=`pwd`/private-benches.sh
    ;;
    *) 
        [ -d $CACHEDIR ] || mkdir $CACHEDIR
        PATH=$PATH:/usr/local/bin # for aws command on darwin
        aws s3 sync s3://tract-ci-builds/model $CACHEDIR
        (cd $CACHEDIR
            [ -d en_libri_real ] || tar zxf en_libri_real.tar.gz
            [ -d en_tdnn_lstm_bn_q7 ] || tar zxf en_tdnn_lstm_bn_q7.tar.gz
        )
        PRIVATE=$CACHEDIR/private/private-benches.sh
    ;;
esac



touch metrics
if [ -e sizes ]
then
    cat sizes >> metrics
fi

if [ $(uname) = "Linux" ]
then
    if [ -r /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor -a `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor` = "userspace" ]
    then
            F=$(printf "%s\n" `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies` | sort -n | tail -1)
            echo $F > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
    fi
fi

net_bench() {
    net=$1
    pb=$2
    shift 2

    $TRACT "$@" --machine-friendly -O bench --allow-random-input $BENCH_OPTS > tract.out
    v=`cat tract.out | grep -a real | cut -f 2 -d ' ' | sed 's/\([0-9]\{9,9\}\)[0-9]*/\1/'`
    echo net.$net.evaltime.$pb $v >> metrics

    $TRACT "$@" --readings --readings-heartbeat 1000 --machine-friendly -O bench --allow-random-input $BENCH_OPTS > tract.out

    for stage in model_ready before_optimize
    do
        pattern=$(echo $stage | sed 's/[_-]/./g')
        v=$(grep -a $pattern readings.out | sed 's/  */ /g;s/^  *//' | cut -f 1 -d ' ')
        echo net.$net.time_to_$stage.$pb $v >> metrics
        v=$(grep -a $pattern readings.out | sed 's/  */ /g;s/^  *//' | cut -f 4 -d ' ')
        echo net.$net.rsz_at_$stage.$pb $v >> metrics
        f=$(grep -a $pattern readings.out | sed 's/  */ /g;s/^  *//' | cut -f 11 -d ' ')
        a=$(grep -a $pattern readings.out | sed 's/  */ /g;s/^  *//' | cut -f 10 -d ' ')
        echo net.$net.active_at_$stage.$pb $(($a-$f)) >> metrics
    done
}

llm_bench() {
    net=$1
    pb=$2
    shift 2

    if  $TRACT "$@" --nnef-tract-core --nnef-tract-transformers -t transformers-detect-all --machine-friendly -O llm-bench $BENCH_OPTS > tract.out
    then
        cat tract.out
        echo llm.$net.pp512.$pb $(cat tract.out | grep -a PP512 | cut -f 2 -d ' ') >> metrics
        echo llm.$net.tg128.$pb $(cat tract.out | grep -a TG128 | cut -f 2 -d ' ') >> metrics
    fi 

    if $TRACT "$@" --readings --readings-heartbeat 1000 --nnef-tract-core --nnef-tract-transformers -t transformers-detect-all --machine-friendly -O llm-bench $BENCH_OPTS > /dev/null
    then
        for stage in model_ready before_optimize
        do
            pattern=$(echo $stage | sed 's/[_-]/./g')
            v=$(grep -a $pattern readings.out | sed 's/  */ /g;s/^  *//' | cut -f 1 -d ' ')
            echo llm.$net.time_to_$stage.$pb $v >> metrics
            v=$(grep -a $pattern readings.out | sed 's/  */ /g;s/^  *//' | cut -f 4 -d ' ')
            echo llm.$net.rsz_at_$stage.$pb $v >> metrics
            f=$(grep -a $pattern readings.out | sed 's/  */ /g;s/^  *//' | cut -f 11 -d ' ')
            a=$(grep -a $pattern readings.out | sed 's/  */ /g;s/^  *//' | cut -f 10 -d ' ')
            if [ -n "$a" -a -n "$f" ]
            then
                 echo llm.$net.active_at_$stage.$pb $(($a-$f)) >> metrics
            fi
        done
    fi
}

net_bench arm_ml_kws_cnn_m pass $CACHEDIR/ARM-ML-KWS-CNN-M.pb -i 49,10,f32 --partial --input-node Mfcc

net_bench hey_snips_v1 400ms $CACHEDIR/hey_snips_v1.pb -i 80,40,f32
net_bench hey_snips_v31 400ms $CACHEDIR/hey_snips_v3.1.pb -i 40,40,f32

net_bench hey_snips_v4_model17 2sec $CACHEDIR/hey_snips_v4_model17.pb -i 200,20,f32
net_bench hey_snips_v4_model17 pulse8 $CACHEDIR/hey_snips_v4_model17.pb -i S,20,f32 --pulse 8
net_bench hey_snips_v4_model17_nnef pulse8 --nnef-tract-pulse $CACHEDIR/hey_snips_v4_model17.alpha1.tar

net_bench mobilenet_v1_1 pass $CACHEDIR/mobilenet_v1_1.0_224_frozen.pb -i 1,224,224,3,f32
net_bench mobilenet_v2_1 pass $CACHEDIR/mobilenet_v2_1.4_224_frozen.pb -i 1,224,224,3,f32

net_bench inceptionv1q pass $CACHEDIR/inceptionv1_quant.nnef.tar.gz --nnef-tract-core
net_bench inceptionv3 pass $CACHEDIR/inception_v3_2016_08_28_frozen.pb -i 1,299,299,3,f32


net_bench mdl-en-2019-Q3-librispeech_onnx 2600ms $CACHEDIR/en_libri_real/model.onnx --output-node output -i 264,40
net_bench mdl-en-2019-Q3-librispeech_onnx pulse_240ms $CACHEDIR/en_libri_real/model.onnx --output-node output -i S,40 --pulse 24
net_bench en_tdnn_lstm_bn_q7 2600ms $CACHEDIR/en_tdnn_lstm_bn_q7/model.onnx --output-node output -i 264,40
net_bench en_tdnn_lstm_bn_q7 pulse_240ms $CACHEDIR/en_tdnn_lstm_bn_q7/model.onnx --output-node output -i S,40 --pulse 24
net_bench en_tdnn_8M 2600ms $CACHEDIR/mdl-en-2019-12-24-aho-corasick-18h01m33s.onnx --output-node output -i 264,40
net_bench en_tdnn_8M pulse_240ms $CACHEDIR/mdl-en-2019-12-24-aho-corasick-18h01m33s.onnx --output-node output -i S,40 --pulse 24
net_bench en_tdnn_8M pulse_180ms $CACHEDIR/mdl-en-2019-12-24-aho-corasick-18h01m33s.onnx --output-node output -i S,40 --pulse 18
net_bench en_tdnn_8M pulse_120ms $CACHEDIR/mdl-en-2019-12-24-aho-corasick-18h01m33s.onnx --output-node output -i S,40 --pulse 12
net_bench en_tdnn_8M_nnef pulse_240ms $CACHEDIR/mdl-en-2019-12-24-aho-corasick-18h01m33s.alpha1.a.tar --nnef-tract-pulse
net_bench en_tdnn_15M 2600ms $CACHEDIR/en_tdnn_15M.onnx --output-node output -i 264,40
net_bench en_tdnn_15M pulse_240ms $CACHEDIR/en_tdnn_15M.onnx --output-node output -i S,40 --pulse 24
net_bench en_tdnn_15M pulse_120ms $CACHEDIR/en_tdnn_15M.onnx --output-node output -i S,40 --pulse 12
net_bench en_tdnn_15M_nnef pulse_240ms $CACHEDIR/en_tdnn_15M.alpha1.tar --nnef-tract-pulse
net_bench dummy-conmer-12M pulse_120ms $CACHEDIR/dummy-conmer-12M.nnef.tar --nnef-tract-core --pulse 12

net_bench en_tdnn_pyt_15M pulse_120ms $CACHEDIR/mdl-en-2023-03-27-allen-17h11m50s.nnef.tar --nnef-tract-core --pulse 12

net_bench speaker_id pulse8 $CACHEDIR/speaker-id-2019-03.onnx -i 1,S,40,f32 --output-node 257 --partial --pulse 8

net_bench voicecom_fake_quant 2sec $CACHEDIR/snips-voice-commands-cnn-fake-quant.pb -i 200,10,f32
net_bench voicecom_float 2sec $CACHEDIR/snips-voice-commands-cnn-float.pb -i 200,10,f32

net_bench trunet pulse1_f32 $CACHEDIR/trunet_dummy.nnef.tgz --nnef-tract-core --pulse 1
net_bench trunet pulse1_f16 $CACHEDIR/trunet_dummy.nnef.tgz --nnef-tract-core --half-floats --pulse 1

. $PRIVATE

if [ $(uname) = "Darwin" ]
then
    LLM_BACKENDS="cpu metal"
fi

if which nvidia-smi 
then 
    LLM_BACKENDS="cpu cuda"
fi

if [ -n "$LLM_BACKENDS" ]
then
    for backend in $LLM_BACKENDS
    do
        case $backend in
            cpu) extra="";;
            metal) extra="--metal"
                   BENCH_OPTS="--warmup-loops 1"
                   ;;
            cuda) extra="--cuda"
                  BENCH_OPTS="--warmup-loops 1"
                  ;;
        esac
        llm_bench llama-3_2-1B-q40ef32-516 $backend $CACHEDIR/Llama-3.2-1B-q40ef32.516.nnef.tgz $extra
        llm_bench openelm-270M-q40ef16-516 $backend $CACHEDIR/OpenELM-270M-q40ef16.516.nnef.tgz $extra
        llm_bench llama-3_2-1B-instruct-q40ef16-541 $backend $CACHEDIR/Llama-3.2-1B-Instruct-q40ef16.541.nnef.tgz $extra
        llm_bench openelm-270M-q40ef16-541 $backend $CACHEDIR/OpenELM-270M-q40ef16.541.nnef.tgz $extra

        if [ "$backend" != "cpu" ]
        then
            llm_bench llama-3_2-3B-q40ef32-516 $backend $CACHEDIR/Llama-3.2-3B-q40ef32.516.nnef.tgz $extra
            llm_bench llama-3_1-8B-instruct-q40ef16-541 $backend $CACHEDIR/Llama-3.1-8B-Instruct-q40ef16.541.nnef.tgz $extra
            llm_bench llama-3_2-3B-instruct-q40ef16-541 $backend $CACHEDIR/Llama-3.2-3B-Instruct-q40ef16.541.nnef.tgz $extra
            llm_bench qwen3-1_7B-q40ef16-541 $backend $CACHEDIR/Qwen3-1.7B-q40ef16.541.nnef.tgz $extra
        fi
    done
fi

end=$(date +%s)

echo bundle.bench-runtime  $(($end - $start)) >> metrics

