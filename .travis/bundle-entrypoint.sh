#!/bin/sh

set -ex

ROOT=`pwd`
CACHEDIR=${CACHEDIR:-$HOME/.cache}
if [ -x tract ]
then
    TRACT=./tract
elif [ -x ./target/release/tract ]
then
    TRACT=./target/release/tract
else
    TRACT="cargo run -p tract -q --release --"
fi

[ -e vars ] && . ./vars

[ -d $CACHEDIR ] || mkdir $CACHEDIR
aws s3 sync s3://tract-ci-builds/model $CACHEDIR
(cd $CACHEDIR
    chmod +x tflite*
    [ -d en_libri_real ] || tar zxf en_libri_real.tar.gz
    [ -d en_tdnn_lstm_bn_q7 ] || tar zxf en_tdnn_lstm_bn_q7.tar.gz
)

touch metrics
if [ -e sizes ]
then
    cat sizes >> metrics
fi

if [ -e benches ]
then
    mkdir -p target/criterion
    for bench in benches/*
    do
        $bench --bench
    done
    for bench in `find target/criterion -path "*/new/*" -name raw.csv`
    do
        group=`cat $bench | tail -1 | cut -d , -f 1 | cut -d . -f 1`
        func=`cat $bench | tail -1 | cut -d , -f 2 | cut -d . -f 1`
        value=`cat $bench | tail -1 | cut -d , -f 3 | cut -d . -f 1`
        nanos=`cat $bench | tail -1 | cut -d , -f 6 | cut -d . -f 1`
        iter=`cat $bench | tail -1 | cut -d , -f 8 | cut -d . -f 1`
        time=$((nanos/iter))
        echo microbench.${group}.${func:-none}.${value:-none} $(($nanos/$iter)) >> metrics
    done
fi

TAIL="--readings --machine-friendly -O profile --bench"

net_bench() {
    net=$1
    pb=$2
    shift 2
    $TRACT "$@" $TAIL > tract.out
    v=`cat tract.out | grep real | cut -f 2 -d ' ' | sed 's/\([0-9]\{9,9\}\)[0-9]*/\1/'`
    echo net.$net.evaltime.$pb $v >> metrics
    v=$(grep model_ready readings.out | sed 's/ \+/ /g;s/^  *//' | cut -f 1 -d ' ')
    echo net.$net.time_to_model_ready.$pb $v >> metrics
    v=$(grep model_ready readings.out | sed 's/ \+/ /g;s/^  *//' | cut -f 4 -d ' ')
    echo net.$net.rsz_at_model_ready.$pb $v >> metrics
    f=$(grep model_ready readings.out | sed 's/ \+/ /g;s/^  *//' | cut -f 11 -d ' ')
    a=$(grep model_ready readings.out | sed 's/ \+/ /g;s/^  *//' | cut -f 10 -d ' ')
    echo net.$net.active_at_model_ready.$pb $(($a-$f)) >> metrics
}

net_bench arm_ml_kws_cnn_m pass $CACHEDIR/ARM-ML-KWS-CNN-M.pb -i 49x10xf32 --partial --input-node Mfcc

net_bench deepspeech_0_4_1 pass \
    $CACHEDIR/deepspeech-0.4.1.pb \
    --input-node input_node -i 1x16x19x26xf32 \
    --input-node input_lengths -i 1xi32=16 --const-input input_lengths \
    --tf-initializer-output-node initialize_state

net_bench hey_snips_v1 400ms $CACHEDIR/hey_snips_v1.pb -i 80x40xf32
net_bench hey_snips_v31 400ms $CACHEDIR/hey_snips_v3.1.pb -i 40x40xf32

net_bench hey_snips_v4_model17 2sec $CACHEDIR/hey_snips_v4_model17.pb -i 200x20xf32
net_bench hey_snips_v4_model17 pulse8 $CACHEDIR/hey_snips_v4_model17.pb -i Sx20xf32 --pulse 8

net_bench mobilenet_v1_1 pass $CACHEDIR/mobilenet_v1_1.0_224_frozen.pb -i 1x224x224x3xf32
net_bench mobilenet_v2_1 pass $CACHEDIR/mobilenet_v2_1.4_224_frozen.pb -i 1x224x224x3xf32

net_bench inceptionv3 pass $CACHEDIR/inception_v3_2016_08_28_frozen.pb -i 1x299x299x3xf32

net_bench kaldi_librispeech_clean_tdnn_lstm_1e_256 2600ms \
    $CACHEDIR/en_libri_real/model.raw -f kaldi --output-node output \
    --kaldi-downsample 3 --kaldi-left-context 5 --kaldi-right-context 15 --kaldi-adjust-final-offset -5 \
    -i 264x40

net_bench kaldi_librispeech_clean_tdnn_lstm_1e_256 pulse_240ms \
    $CACHEDIR/en_libri_real/model.raw -f kaldi  --output-node output \
    --kaldi-downsample 3 --kaldi-left-context 5 --kaldi-right-context 15 --kaldi-adjust-final-offset -5 \
    -i Sx40 --pulse 24 \

net_bench mdl-en-2019-Q3-librispeech_onnx 2600ms $CACHEDIR/en_libri_real/model.onnx --output-node output -i 264x40
net_bench mdl-en-2019-Q3-librispeech_onnx pulse_240ms $CACHEDIR/en_libri_real/model.onnx --output-node output -i Sx40 --pulse 24
net_bench en_tdnn_lstm_bn_q7 2600ms $CACHEDIR/en_tdnn_lstm_bn_q7/model.onnx --output-node output -i 264x40
net_bench en_tdnn_lstm_bn_q7 pulse_240ms $CACHEDIR/en_tdnn_lstm_bn_q7/model.onnx --output-node output -i Sx40 --pulse 24
net_bench en_tdnn_8M 2600ms $CACHEDIR/mdl-en-2019-12-24-aho-corasick-18h01m33s.onnx --output-node output -i 264x40
net_bench en_tdnn_8M pulse_240ms $CACHEDIR/mdl-en-2019-12-24-aho-corasick-18h01m33s.onnx --output-node output -i Sx40 --pulse 24

net_bench speaker_id pulse8 $CACHEDIR/speaker-id-2019-03.onnx -i 1xSx40xf32 --output-node 257 --partial --pulse 8

net_bench voicecom_fake_quant 2sec $CACHEDIR/snips-voice-commands-cnn-fake-quant.pb -i 200x10xf32
net_bench voicecom_float 2sec $CACHEDIR/snips-voice-commands-cnn-float.pb -i 200x10xf32

if [ -e /etc/issue ] && ( cat /etc/issue | grep Raspbian )
then
    cpu=`awk '/^Revision/ {sub("^1000", "", $3); print $3}' /proc/cpuinfo`
    # raspi 3 can run official tflite builds
    if [ "$cpu" = "a22082" ]
    then
        tflites="rpitools official_rpi rpitools_2019_03 official_rpi_2019_03"
    else
        tflites="rpitools rpitools_2019_03"
    fi
elif [ -e /etc/issue ] && ( cat /etc/issue | grep i.MX )
then
    if [ `uname -m` = "aarch64" ]
    then
        tflites="aarch64_unknown_linux_gnu aarch64_unknown_linux_gnu_2019_03"
    elif [ `uname -m` = "armv7l" ]
    then
        tflites="official_rpi official_rpi_2019_03"
    fi
elif [ -e /etc/issue ] && ( cat /etc/issue | grep buster )
then
    if [ `uname -m` = "aarch64" ]
    then
        tflites="aarch64_unknown_linux_gnu aarch64_unknown_linux_gnu_2019_03"
    fi
fi

for tflite in $tflites
do
    $CACHEDIR/tflite_benchmark_model_$tflite \
        --graph=$CACHEDIR/inception_v3_2016_08_28_frozen.tflite \
        --num_runs=1 \
    2> bench
    usec=`cat bench | tail -1 | sed "s/.* //"`
    sec=`python -c "print(float($usec) / 1000000)"`
    echo net.inceptionv3.tflite_$tflite.pass $sec >> metrics

    $CACHEDIR/tflite_benchmark_model_$tflite \
        --graph=$CACHEDIR/hey_snips_v1.tflite \
    2> bench
    usec=`cat bench | tail -1 | sed "s/.* //"`
    sec=`python -c "print(float($usec) / 1000000)"`
    echo net.hey_snips_v1.tflite_$tflite.400ms $sec >> metrics

    $CACHEDIR/tflite_benchmark_model_$tflite \
        --graph=$CACHEDIR/ARM-ML-KWS-CNN-M.tflite \
    2> bench
    usec=`cat bench | tail -1 | sed "s/.* //"`
    sec=`python -c "print(float($usec) / 1000000)"`
    echo net.arm_ml_kws_cnn_m.tflite_$tflite.pass $sec >> metrics

    $CACHEDIR/tflite_benchmark_model_$tflite \
        --graph=$CACHEDIR/mobilenet_v1_1.0_224.tflite \
        --num_runs=1 \
    2> bench
    usec=`cat bench | tail -1 | sed "s/.* //"`
    sec=`python -c "print(float($usec) / 1000000)"`
    echo net.mobilenet_v1.tflite_$tflite.pass $sec >> metrics

    $CACHEDIR/tflite_benchmark_model_$tflite \
        --graph=$CACHEDIR/mobilenet_v2_1.4_224.tflite \
        --num_runs=1 \
    2> bench
    usec=`cat bench | tail -1 | sed "s/.* //"`
    sec=`python -c "print(float($usec) / 1000000)"`
    echo net.mobilenet_v2.tflite_$tflite.pass $sec >> metrics
done
