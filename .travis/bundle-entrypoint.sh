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
        group=`cat $bench | tail -1 | cut -d , -f 1`
        nanos=`cat $bench | tail -1 | cut -d , -f 4`
        iter=`cat $bench | tail -1 | cut -d , -f 5`
        time=$((nanos/iter))
        echo microbench.${group}.${func:-none}.${value:-none} $(($nanos/$iter)) >> metrics
    done
)

inceptionv3=`$TRACT --machine-friendly $CACHEDIR/inception_v3_2016_08_28_frozen.pb \
    -O -i 1x299x299x3xf32 profile --bench \
    | grep real | cut -f 2 -d ' '`

voicecom_float=`$TRACT --machine-friendly $CACHEDIR/snips-voice-commands-cnn-float.pb \
    -O -i 200x10xf32 profile --bench \
    | grep real | cut -f 2 -d ' '`

voicecom_fake_quant=`$TRACT --machine-friendly $CACHEDIR/snips-voice-commands-cnn-fake-quant.pb \
    -O -i 200x10xf32 profile --bench \
    | grep real | cut -f 2 -d ' '`

arm_ml_kws_cnn_m=`$TRACT --machine-friendly $CACHEDIR/ARM-ML-KWS-CNN-M.pb \
    -O -i 49x10xf32 --input-node Mfcc profile --bench \
    | grep real | cut -f 2 -d ' '`

hey_snips_v31_400ms=`$TRACT --machine-friendly $CACHEDIR/hey_snips_v3.1.pb \
    -O -i 40x40xf32 profile --bench \
    | grep real | cut -f 2 -d ' '`

hey_snips_v4_model17_2sec=`$TRACT --machine-friendly $CACHEDIR/hey_snips_v4_model17.pb \
    -O -i 200x20xf32 profile --bench \
    | grep real | cut -f 2 -d ' '`

hey_snips_v4_model17_pulse8=`$TRACT --machine-friendly $CACHEDIR/hey_snips_v4_model17.pb \
    -O -i Sx20xf32 --pulse 8 profile --bench \
    | grep real | cut -f 2 -d ' '`

echo net.inceptionv3.evaltime.pass $inceptionv3 >> metrics
echo net.arm_ml_kws_cnn_m.evaltime.pass $arm_ml_kws_cnn_m >> metrics
echo net.voicecom_float.evaltime.2sec $voicecom_float >> metrics
echo net.voicecom_fake_quant.evaltime.2sec $voicecom_fake_quant >> metrics
echo net.hey_snips_v31.evaltime.400ms $hey_snips_v31_400ms >> metrics
echo net.hey_snips_v4_model17.evaltime.2sec $hey_snips_v4_model17_2sec >> metrics
echo net.hey_snips_v4_model17.evaltime.pulse8 $hey_snips_v4_model17_pulse8 >> metrics

if [ -e /etc/issue ] && ( cat /etc/issue | grep Raspbian )
then
    cpu=`awk '/^Revision/ {sub("^1000", "", $3); print $3}' /proc/cpuinfo`
    # raspi 3 can run official tflite builds
    if [ "$cpu" = "a22082" ]
    then
        tflites="rpitools official_rpi"
    else
        tflites=rpitools
    fi
elif [ -e /etc/issue ] && ( cat /etc/issue | grep i.MX )
then
    if [ `uname -m` = "aarch64" ]
    then
        tflites=aarch64
    fi
fi

for tflite in $tflites
do
    $CACHEDIR/tflite_benchmark_model_$tflite \
        --graph=$CACHEDIR/inception_v3_2016_08_28_frozen.tflite \
        --num_runs=1 \
    2> bench
    usec=`cat bench | grep 'curr=' | tail -1 | sed "s/.*=//"`
    sec=`echo "scale=6; $usec / 1000000" | bc -l`
    echo net.inceptionv3.tflite_$tflite.pass $sec >> metrics

    $CACHEDIR/tflite_benchmark_model_$tflite \
        --graph=$CACHEDIR/hey_snips_v3.1.tflite \
    2> bench
    usec=`cat bench | grep 'curr=' | tail -1 | sed "s/.*=//"`
    sec=`echo "scale=6; $usec / 1000000" | bc -l`
    echo net.hey_snips_v31.tflite_$tflite.400ms $sec >> metrics
done

