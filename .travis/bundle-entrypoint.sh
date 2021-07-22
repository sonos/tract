#!/bin/sh

set -ex

start=$(date +%s)

ROOT=`pwd`

if [ -x tract ]
then
    TRACT=./tract
else
    cargo build -p tract -q --release
    TRACT=./target/release/tract
fi

CACHEDIR=${CACHEDIR:-$HOME/.cache}
case $CACHEDIR in
  "http"*) ;;
  *) 
      [ -d $CACHEDIR ] || mkdir $CACHEDIR
      aws s3 sync s3://tract-ci-builds/model $CACHEDIR
      (cd $CACHEDIR
          [ -d en_libri_real ] || tar zxf en_libri_real.tar.gz
          [ -d en_tdnn_lstm_bn_q7 ] || tar zxf en_tdnn_lstm_bn_q7.tar.gz
      )
    ;;
esac



touch metrics
if [ -e sizes ]
then
    cat sizes >> metrics
fi

net_bench() {
    net=$1
    pb=$2
    shift 2

    $TRACT "$@" --machine-friendly -O bench $BENCH_OPTS > tract.out
    v=`cat tract.out | grep real | cut -f 2 -d ' ' | sed 's/\([0-9]\{9,9\}\)[0-9]*/\1/'`
    echo net.$net.evaltime.$pb $v >> metrics

    $TRACT "$@" --readings --readings-heartbeat 1000 --machine-friendly -O bench $BENCH_OPTS > tract.out

    for stage in model_ready before_optimize
    do
        pattern=$(echo $stage | sed 's/[_-]/./g')
        v=$(grep $pattern readings.out | sed 's/  */ /g;s/^  *//' | cut -f 1 -d ' ')
        echo net.$net.time_to_$stage.$pb $v >> metrics
        v=$(grep $pattern readings.out | sed 's/  */ /g;s/^  *//' | cut -f 4 -d ' ')
        echo net.$net.rsz_at_$stage.$pb $v >> metrics
        f=$(grep $pattern readings.out | sed 's/  */ /g;s/^  *//' | cut -f 11 -d ' ')
        a=$(grep $pattern readings.out | sed 's/  */ /g;s/^  *//' | cut -f 10 -d ' ')
        echo net.$net.active_at_$stage.$pb $(($a-$f)) >> metrics
    done
}

mem=$(free -m | grep Mem | awk '{ print $2 }')

# if [ $mem -gt 600 ]
# then
#     net_bench deepspeech_0_4_1 pass \
#         $CACHEDIR/deepspeech-0.4.1.pb \
#         --input-node input_node -i 1,16,19,26,f32 \
#         --input-node input_lengths -i 1,i32=16 --const-input input_lengths \
#         --tf-initializer-output-node initialize_state
# fi

net_bench arm_ml_kws_cnn_m pass $CACHEDIR/ARM-ML-KWS-CNN-M.pb -i 49,10,f32 --partial --input-node Mfcc

net_bench hey_snips_v1 400ms $CACHEDIR/hey_snips_v1.pb -i 80,40,f32
net_bench hey_snips_v31 400ms $CACHEDIR/hey_snips_v3.1.pb -i 40,40,f32

net_bench hey_snips_v4_model17 2sec $CACHEDIR/hey_snips_v4_model17.pb -i 200,20,f32
net_bench hey_snips_v4_model17 pulse8 $CACHEDIR/hey_snips_v4_model17.pb -i S,20,f32 --pulse 8
net_bench hey_snips_v4_model17_nnef pulse8 --nnef-tract-pulse $CACHEDIR/hey_snips_v4_model17.alpha1.tar

net_bench mobilenet_v1_1 pass $CACHEDIR/mobilenet_v1_1.0_224_frozen.pb -i 1,224,224,3,f32
net_bench mobilenet_v2_1 pass $CACHEDIR/mobilenet_v2_1.4_224_frozen.pb -i 1,224,224,3,f32

net_bench inceptionv3 pass $CACHEDIR/inception_v3_2016_08_28_frozen.pb -i 1,299,299,3,f32

net_bench kaldi_librispeech_clean_tdnn_lstm_1e_256 2600ms \
    $CACHEDIR/en_libri_real/model.raw -f kaldi --output-node output \
    --kaldi-downsample 3 --kaldi-left-context 5 --kaldi-right-context 15 --kaldi-adjust-final-offset -5 \
    -i 264,40

net_bench kaldi_librispeech_clean_tdnn_lstm_1e_256 pulse_240ms \
    $CACHEDIR/en_libri_real/model.raw -f kaldi  --output-node output \
    --kaldi-downsample 3 --kaldi-left-context 5 --kaldi-right-context 15 --kaldi-adjust-final-offset -5 \
    -i S,40 --pulse 24 \

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

net_bench speaker_id pulse8 $CACHEDIR/speaker-id-2019-03.onnx -i 1,S,40,f32 --output-node 257 --partial --pulse 8

net_bench voicecom_fake_quant 2sec $CACHEDIR/snips-voice-commands-cnn-fake-quant.pb -i 200,10,f32
net_bench voicecom_float 2sec $CACHEDIR/snips-voice-commands-cnn-float.pb -i 200,10,f32

end=$(date +%s)

echo bundle.bench-runtime  $(($end - $start)) >> metrics
