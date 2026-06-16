#!/bin/sh

set -e

# Foldable log sections: real groups under GitHub Actions, plain headers elsewhere.
# This script also runs on busybox devices, where the ::group:: commands are just noise.
group() {
    if [ -n "$GITHUB_ACTIONS" ]; then printf '::group::%s\n' "$1"; else printf '== %s ==\n' "$1"; fi
}
endgroup() {
    if [ -n "$GITHUB_ACTIONS" ]; then printf '::endgroup::\n'; fi
}

start=$(date +%s)

ROOT=$(pwd)

if [ -n "$TRACT_RUN" ]; then
    TRACT=$TRACT_RUN
elif [ -x tract ]; then
    TRACT="./tract"
else
    group "build tract-cli"
    cargo build -p tract-cli -q --release
    endgroup
    TRACT="./target/release/tract"
fi

group "fetch models"
CACHEDIR=${CACHEDIR:-$HOME/.cache/tract-ci-minion-models}
[ -d "$CACHEDIR" ] || mkdir "$CACHEDIR"
PATH=$PATH:/usr/local/bin # for aws command on darwin
aws s3 sync s3://tract-ci-builds/model "$CACHEDIR" --only-show-errors || echo "Warning: aws s3 sync failed, continuing with cached models"
(cd "$CACHEDIR"
    [ -d en_libri_real ] || tar zxf en_libri_real.tar.gz
    [ -d en_tdnn_lstm_bn_q7 ] || tar zxf en_tdnn_lstm_bn_q7.tar.gz
)
endgroup

touch metrics
[ -e sizes ] && cat sizes >> metrics

if [ "$(uname)" = "Linux" ] && [ -r /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ] && [ "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)" = "userspace" ]; then
    F=$(printf '%s\n' $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies) | sort -n | tail -1)
    echo "$F" > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
fi

# Expectation-guided retry: with an expectations file (EXPECTATIONS, 'metric expected
# threshold' lines from bench-data history), re-run a bench whose measured value moved
# worse-than-expected by at least its threshold — i.e. far enough to show as a PR red —
# and keep the best per metric. Same bar as the report's reds, so every red has been
# measured RETRY_MAX+1 times; a real change reproduces (best-of still reports it), a
# one-shot glitch does not. No expectations (fresh host / legacy minion) -> single shot,
# unchanged.
RETRY_MAX=${RETRY_MAX:-2}    # re-runs on a miss; total tries = RETRY_MAX + 1

# Canonicalize '-' -> '_' at the source so metric keys match the expectations and
# bench-data (the old minion did this via `tr`); keeps the awk match free of gsub,
# which the macOS BWK awk handled differently from gawk/mawk/busybox.
emit() { printf '%s %s\n' "$(echo "$1" | sed 's/-/_/g')" "$2" >> "$CUR"; }
newtmp() { mktemp "${TMPDIR:-/tmp}/bench.XXXXXX"; }   # explicit template: busybox mktemp wants one

out_of_threshold() {  # true if some metric in file $1 moved worse-than-expected by >= its threshold
    [ -n "$EXPECTATIONS" ] || return 1
    awk '
        FNR == NR { E[$1] = $2 + 0; T[$1] = $3 + 0; next }
        { v = $2 + 0
          if (($1 in E) && E[$1] > 0) {
              pct = (v - E[$1]) / E[$1] * 100
              if ($1 ~ /\.(pp|tg)[0-9]+\./) worse = -pct; else worse = pct
              if (worse >= T[$1]) bad = 1
          } }
        END { if (bad) exit 0; exit 1 }
    ' "$EXPECTATIONS" "$1"
}

merge_best() {  # $1 <- per-metric best of $1 and $2 (min, or max for pp/tg throughput)
    out=$(newtmp)
    awk '
        FNR == NR { b[$1] = $2; next }
        { if (!($1 in b)) { b[$1] = $2; next }
          v = $2 + 0; bv = b[$1] + 0
          hb = ($1 ~ /\.(pp|tg)[0-9]+\./)
          if (hb) { if (v > bv) b[$1] = $2 }
          else    { if (v < bv) b[$1] = $2 } }
        END { for (k in b) print k, b[k] }
    ' "$1" "$2" > "$out"
    mv "$out" "$1"
}

bench_run() {  # bench_run <measure-fn> <args...>
    fn=$1
    shift
    if [ ! -s "$EXPECTATIONS" ]; then CUR=metrics; "$fn" "$@"; return; fi
    best=$(newtmp); CUR=$best; : > "$best"; "$fn" "$@"
    tries=0
    while [ "$tries" -lt "$RETRY_MAX" ] && out_of_threshold "$best"; do
        tries=$((tries + 1))
        printf '    retry %s (off expectation)\n' "$tries"
        cand=$(newtmp); CUR=$cand; : > "$cand"; "$fn" "$@"
        merge_best "$best" "$cand"
        rm -f "$cand"
    done
    cat "$best" >> metrics
    rm -f "$best"
}

net_bench() { printf '  %s %s\n' "$1" "$2"; bench_run _net_measure "$@"; }
llm_bench() { printf '  %s %s\n' "$1" "$2"; bench_run _llm_measure "$@"; }

_net_measure() {
    net=$1
    pb=$2
    shift 2

    $TRACT "$@" --machine-friendly -O bench --allow-random-input $BENCH_OPTS > tract.out
    v=$(grep -a real tract.out | cut -f 2 -d ' ' | sed 's/\([0-9]\{9,9\}\)[0-9]*/\1/')
    emit net.$net.evaltime.$pb "$v"

    $TRACT "$@" --readings --readings-heartbeat 1000 --machine-friendly -O bench --allow-random-input $BENCH_OPTS > tract.out

    for stage in model_ready before_optimize
    do
        pattern=$(echo $stage | sed 's/[_-]/./g')
        v=$(grep -a $pattern readings.out | sed 's/  */ /g;s/^  *//' | cut -f 1 -d ' ')
        emit net.$net.time_to_$stage.$pb "$v"
        v=$(grep -a $pattern readings.out | sed 's/  */ /g;s/^  *//' | cut -f 4 -d ' ')
        emit net.$net.rsz_at_$stage.$pb "$v"
        f=$(grep -a $pattern readings.out | sed 's/  */ /g;s/^  *//' | cut -f 11 -d ' ')
        a=$(grep -a $pattern readings.out | sed 's/  */ /g;s/^  *//' | cut -f 10 -d ' ')
        emit net.$net.active_at_$stage.$pb "$((a - f))"
    done
}

_llm_measure() {
    net=$1
    pb=$2
    shift 2

    if $TRACT "$@" --llm --machine-friendly -O llm-bench $BENCH_OPTS > tract.out
    then
        emit llm.$net.pp512.$pb "$(grep -a PP512 tract.out | cut -f 2 -d ' ')"
        emit llm.$net.tg128.$pb "$(grep -a TG128 tract.out | cut -f 2 -d ' ')"
    fi

    if $TRACT "$@" --readings --readings-heartbeat 1000 --llm --machine-friendly -O llm-bench $BENCH_OPTS > /dev/null
    then
        for stage in model_ready before_optimize
        do
            pattern=$(echo $stage | sed 's/[_-]/./g')
            v=$(grep -a $pattern readings.out | sed 's/  */ /g;s/^  *//' | cut -f 1 -d ' ')
            emit llm.$net.time_to_$stage.$pb "$v"
            v=$(grep -a $pattern readings.out | sed 's/  */ /g;s/^  *//' | cut -f 4 -d ' ')
            emit llm.$net.rsz_at_$stage.$pb "$v"
            f=$(grep -a $pattern readings.out | sed 's/  */ /g;s/^  *//' | cut -f 11 -d ' ')
            a=$(grep -a $pattern readings.out | sed 's/  */ /g;s/^  *//' | cut -f 10 -d ' ')
            if [ -n "$a" ] && [ -n "$f" ]
            then
                emit llm.$net.active_at_$stage.$pb "$((a - f))"
            fi
        done
    fi
}

group "net benches"
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
net_bench trunet pulse1_f16 $CACHEDIR/trunet_dummy.nnef.tgz --nnef-tract-core -t f32_to_f16 --pulse 1
endgroup

if [ "$(uname)" = "Darwin" ]; then
    LLM_BACKENDS="cpu metal"
fi

if which nvidia-smi > /dev/null 2>&1; then
    LLM_BACKENDS="cpu cuda"
fi

if [ -n "$LLM_BACKENDS" ]; then
    for backend in $LLM_BACKENDS
    do
        group "llm benches: $backend"
        case $backend in
            cpu) extra="--timeout 180";;
            metal) extra="--metal --timeout 60"
                   BENCH_OPTS="--warmup-loops 1"
                   ;;
            cuda) extra="--cuda --timeout 60"
                  BENCH_OPTS="--warmup-loops 1"
                  ;;
        esac
        llm_bench llama-3_2-1B-q40ef32-516 $backend $CACHEDIR/Llama-3.2-1B-q40ef32.516.nnef.tgz $extra
        llm_bench openelm-270M-q40ef16-516 $backend $CACHEDIR/OpenELM-270M-q40ef16.516.nnef.tgz $extra
        llm_bench llama-3_2-1B-instruct-q40ef16-541 $backend $CACHEDIR/Llama-3.2-1B-Instruct-q40ef16.541.nnef.tgz $extra
        llm_bench openelm-270M-q40ef16-541 $backend $CACHEDIR/OpenELM-270M-q40ef16.541.nnef.tgz $extra
        net_bench parakeet-tdt-600m-v3-f32f32-preprocessor_1s $backend $CACHEDIR/parakeet-tdt-0.6b-v3-f32f32.608.preprocessor.nnef.tgz \
                        -t transformers_detect_all --nnef-tract-transformers --set B=1 --set A=16000 $extra
        net_bench parakeet-tdt-600m-v3-f32f32-encoder_1s $backend $CACHEDIR/parakeet-tdt-0.6b-v3-f32f32.608.encoder.p1.nnef.tgz \
                        -t transformers_detect_all --nnef-tract-transformers --set B=1 --set S=100 $extra
        net_bench parakeet-tdt-600m-v3-f32f32-decoder_pass $backend $CACHEDIR/parakeet-tdt-0.6b-v3-f32f32.608.decoder.nnef.tgz \
                        -t transformers_detect_all --nnef-tract-transformers --set B=1 --set T=1 $extra
        net_bench parakeet-tdt-600m-v3-f32f32-joint_pass $backend $CACHEDIR/parakeet-tdt-0.6b-v3-f32f32.608.joint.nnef.tgz \
                        -t transformers_detect_all --nnef-tract-transformers --set B=1 --set R=1 --set U=1 $extra

        if [ "$backend" != "cpu" ]
        then
            llm_bench llama-3_2-3B-q40ef32-516 $backend $CACHEDIR/Llama-3.2-3B-q40ef32.516.nnef.tgz $extra
            llm_bench llama-3_1-8B-instruct-q40ef16-541 $backend $CACHEDIR/Llama-3.1-8B-Instruct-q40ef16.541.nnef.tgz $extra
            llm_bench llama-3_2-3B-instruct-q40ef16-541 $backend $CACHEDIR/Llama-3.2-3B-Instruct-q40ef16.541.nnef.tgz $extra
            llm_bench qwen3-1_7B-q40ef16-541 $backend $CACHEDIR/Qwen3-1.7B-q40ef16.541.nnef.tgz $extra
        fi
        endgroup
    done
fi

end=$(date +%s)

echo bundle.bench_runtime $((end - start)) >> metrics
