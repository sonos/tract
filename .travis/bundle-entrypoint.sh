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

mkdir -p $CACHEDIR

./tract --help

. ./vars

(
    cd $CACHEDIR
    if [ ! -e inception_v3_2016_08_28_frozen.pb.tar.gz ]
    then
        wget http://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz
    fi
    if [ !  -e inception_v3_2016_08_28_frozen.pb ]
    then
        tar zxf inception_v3_2016_08_28_frozen.pb.tar.gz
    fi
    for m in \
        ARM-ML-KWS-CNN-M.pb \
        snips-voice-commands-cnn-float.pb \
        snips-voice-commands-cnn-fake-quant.pb \
        hey_snips_v3.1.pb \
        hey_snips_v4_model17.pb
    do
        [ -e "$m" ] || aws s3 cp s3://tract-ci-builds/model/$m .
    done
)

binary_size_cli=`stat -c "%s" tract`

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


echo binary_size.cli $binary_size_cli > metrics
echo net.inceptionv3.evaltime.pass $inceptionv3 >> metrics
echo net.arm_ml_kws_cnn_m.evaltime.pass $arm_ml_kws_cnn_m >> metrics
echo net.voicecom_float.evaltime.2sec $voicecom_float >> metrics
echo net.voicecom_fake_quant.evaltime.2sec $voicecom_fake_quant >> metrics
echo net.hey_snips_v31.evaltime.400ms $hey_snips_v31_400ms >> metrics
echo net.hey_snips_v4_model17.evaltime.2sec $hey_snips_v4_model17_2sec >> metrics
echo net.hey_snips_v4_model17.evaltime.pulse8 $hey_snips_v4_model17_pulse8 >> metrics
