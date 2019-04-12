#!/bin/sh

set -ex

ROOT=`pwd`
CACHEDIR=${CACHEDIR:-$HOME/.cache}
if ./tract --version
then
    TRACT=./tract
else
    TRACT=./target/release/tract
fi

. ./vars

(cd $CACHEDIR ; aws s3 sync s3://tract-ci-builds/model $CACHEDIR)

chmod +x $CACHEDIR/tflite*

binary_size_cli=`stat -c "%s" tract`
echo binary_size.cli $binary_size_cli > metrics

(
    mkdir -p target/criterion
    for bench in benches/*
    do
        $bench
    done
    for bench in `find target/criterion -path "*/new/*" -name raw.csv`
    do
        group=`cat $bench | tail -1 | cut -d , -f 1 | cut -d . -f 1`
        nanos=`cat $bench | tail -1 | cut -d , -f 4 | cut -d . -f 1`
        iter=`cat $bench | tail -1 | cut -d , -f 5 | cut -d . -f 1`
        time=$((nanos/iter))
        echo microbench.${group}.${func:-none}.${value:-none} $(($nanos/$iter)) >> metrics
    done
)

arm_ml_kws_cnn_m=`$TRACT --machine-friendly $CACHEDIR/ARM-ML-KWS-CNN-M.pb \
    -O -i 49x10xf32 --input-node Mfcc profile --bench \
    | grep real | cut -f 2 -d ' ' | sed 's/\([0-9]\{9,9\}\)[0-9]*/\1/'`
echo net.arm_ml_kws_cnn_m.evaltime.pass $arm_ml_kws_cnn_m >> metrics

deepspeech_0_4_1=`$TRACT --machine-friendly $CACHEDIR/deepspeech-0.4.1.pb \
    --input-node input_node -i 1x16x19x26xf32 \
    --input-node input_lengths -i 1xi32=16 \
    -O profile --bench \
    | grep real | cut -f 2 -d ' ' | sed 's/\([0-9]\{9,9\}\)[0-9]*/\1/'`
echo net.deepspeech_0_4_1.evaltime.pass $deepspeech_0_4_1 >> metrics

hey_snips_v1_400ms=`$TRACT --machine-friendly $CACHEDIR/hey_snips_v1.pb \
    -O -i 41x40xf32 profile --bench \
    | grep real | cut -f 2 -d ' ' | sed 's/\([0-9]\{9,9\}\)[0-9]*/\1/'`
echo net.hey_snips_v1.evaltime.400ms $hey_snips_v1_400ms >> metrics

hey_snips_v31_400ms=`$TRACT --machine-friendly $CACHEDIR/hey_snips_v3.1.pb \
    -O -i 40x40xf32 profile --bench \
    | grep real | cut -f 2 -d ' ' | sed 's/\([0-9]\{9,9\}\)[0-9]*/\1/'`
echo net.hey_snips_v31.evaltime.400ms $hey_snips_v31_400ms >> metrics

hey_snips_v4_model17_2sec=`$TRACT --machine-friendly $CACHEDIR/hey_snips_v4_model17.pb \
    -O -i 200x20xf32 profile --bench \
    | grep real | cut -f 2 -d ' ' | sed 's/\([0-9]\{9,9\}\)[0-9]*/\1/'`
echo net.hey_snips_v4_model17.evaltime.2sec $hey_snips_v4_model17_2sec >> metrics

hey_snips_v4_model17_pulse8=`$TRACT --machine-friendly $CACHEDIR/hey_snips_v4_model17.pb \
    -O -i Sx20xf32 --pulse 8 profile --bench \
    | grep real | cut -f 2 -d ' ' | sed 's/\([0-9]\{9,9\}\)[0-9]*/\1/'`
echo net.hey_snips_v4_model17.evaltime.pulse8 $hey_snips_v4_model17_pulse8 >> metrics

mobilenet_v1_1=`$TRACT --machine-friendly $CACHEDIR/mobilenet_v1_1.0_224_frozen.pb \
    -O -i 1x224x224x3xf32 profile --bench \
    | grep real | cut -f 2 -d ' ' | sed 's/\([0-9]\{9,9\}\)[0-9]*/\1/'`
echo net.mobilenet_v1_1.evaltime.pass $mobilenet_v1_1 >> metrics

mobilenet_v2_1=`$TRACT --machine-friendly $CACHEDIR/mobilenet_v2_1.4_224_frozen.pb \
    -O -i 1x224x224x3xf32 profile --bench \
    | grep real | cut -f 2 -d ' ' | sed 's/\([0-9]\{9,9\}\)[0-9]*/\1/'`
echo net.mobilenet_v2_1.evaltime.pass $mobilenet_v2_1 >> metrics

inceptionv3=`$TRACT --machine-friendly $CACHEDIR/inception_v3_2016_08_28_frozen.pb \
    -O -i 1x299x299x3xf32 profile --bench \
    | grep real | cut -f 2 -d ' ' | sed 's/\([0-9]\{9,9\}\)[0-9]*/\1/'`
echo net.inceptionv3.evaltime.pass $inceptionv3 >> metrics

speaker_id_pulse8=`$TRACT --machine-friendly $CACHEDIR/speaker-id-2019-03.onnx \
    -O -i 1xSx40xf32 --output-node 257 --pulse 8 profile --bench \
    | grep real | cut -f 2 -d ' ' | sed 's/\([0-9]\{9,9\}\)[0-9]*/\1/'`
echo net.speaker_id.evaltime.pulse8 $speaker_id_pulse8 >> metrics

voicecom_fake_quant=`$TRACT --machine-friendly $CACHEDIR/snips-voice-commands-cnn-fake-quant.pb \
    -O -i 200x10xf32 profile --bench \
    | grep real | cut -f 2 -d ' ' | sed 's/\([0-9]\{9,9\}\)[0-9]*/\1/'`
echo net.voicecom_fake_quant.evaltime.2sec $voicecom_fake_quant >> metrics

voicecom_float=`$TRACT --machine-friendly $CACHEDIR/snips-voice-commands-cnn-float.pb \
    -O -i 200x10xf32 profile --bench \
    | grep real | cut -f 2 -d ' ' | sed 's/\([0-9]\{9,9\}\)[0-9]*/\1/'`
echo net.voicecom_float.evaltime.2sec $voicecom_float >> metrics

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

