#!/bin/sh

set -ex

ROOT=$(realpath $(dirname $(realpath $0))/../..)
. $ROOT/.travis/ci-system-setup.sh

MODEL=nvidia--nemotron-speech-streaming-en-0.6b-f32f32
S3DIR=asr/613/$MODEL

for rt in $TRACT_RUNTIMES
do
	gpu_assert=""
	case "$rt" in
		--cuda) gpu_assert="--assert-op-only Cuda*,Gpu*,DeviceSync*,Const,Source,STFT,Pad,IsNan,Add,Range,Cast,Eq,Div,Sub,Scan,Gather";;
		--metal) gpu_assert="--assert-op-only Metal*,Gpu*,DeviceSync*,Const,Source,STFT,Pad,IsNan,Add,Range,Cast,Eq,Div,Sub,Scan,Gather,Reduce*";;
	esac

	for m in preprocessor encoder decoder joint
	do
		# Encoder uses a patched model with upper bound assertion on AUDIO_SIGNAL__TIME
		if [ "$m" = "encoder" ]; then
			nnef_file=$MODEL.$m.p1.nnef.tgz
		else
			nnef_file=$MODEL.$m.nnef.tgz
		fi
		$CACHE_FILE \
			$S3DIR/$nnef_file \
			$S3DIR/$MODEL.$m.io.npz

		$TRACT_RUN $MODELS/$S3DIR/$nnef_file $rt --nnef-tract-transformers -t transformers_detect_all run \
			--input-from-bundle $MODELS/$S3DIR/$MODEL.$m.io.npz --assert-output-bundle $MODELS/$S3DIR/$MODEL.$m.io.npz \
			--approx very $gpu_assert
	done
done

model_prefix=$MODELS/$S3DIR/$MODEL

# Check that the patch transform eliminates all Iff nodes,
# and that select_outputs can reduce the model to a single output
$TRACT_RUN $model_prefix.preprocessor.nnef.tgz \
	-t 'concretize_symbols(values: {"BATCH": 1})' \
	-t 'patch(body: "length = tract_core_shape_of(input_signal)[1];")' \
	-t 'select_outputs(outputs: ["processed_signal"])' \
	dump -q \
	--assert-op-count Iff 0

# Check that the preprocessor can be pulsified (both large and small pulse)
$TRACT_RUN $model_prefix.preprocessor.nnef.tgz \
	-t 'concretize_symbols(values: {"BATCH": 1})' \
	-t 'patch(body: "length = tract_core_shape_of(input_signal)[1];")' \
	-t 'select_outputs(outputs: ["processed_signal"])' \
	-t 'pulse(symbol: Some("INPUT_SIGNAL__TIME"), pulse: "4800")' \
	dump -q
$TRACT_RUN $model_prefix.preprocessor.nnef.tgz \
	-t 'concretize_symbols(values: {"BATCH": 1})' \
	-t 'patch(body: "length = tract_core_shape_of(input_signal)[1];")' \
	-t 'select_outputs(outputs: ["processed_signal"])' \
	-t 'pulse(symbol: Some("INPUT_SIGNAL__TIME"), pulse: "1600")' \
	dump -q

# Check that the encoder can be pulsified.
# The encoder subsamples by 8x (three stride-2 convolutions) before the transformer.
# The chunk-window mask has P=14 transformer tokens per chunk, so the input pulse
# must be 14 * 8 = 112 audio frames.
$TRACT_RUN $model_prefix.encoder.p1.nnef.tgz \
	--nnef-tract-transformers \
	-t 'pulse(symbol: Some("AUDIO_SIGNAL__TIME"), pulse: "112")' \
	dump -q

# Check that the pulsified encoder runs without error.
# We don't assert output equality because the test audio (744 frames) is not
# exactly divisible by the pulse (112), causing small mismatches at the tail
# of the last partial pulse.  The batch test above covers numerical correctness;
# this test verifies the pulsified model executes end-to-end.
$TRACT_RUN $model_prefix.encoder.p1.nnef.tgz \
	--nnef-tract-transformers \
	-t 'concretize_symbols(values: {"BATCH": 1})' \
	-t 'patch(body: "length = tract_core_shape_of(audio_signal)[2];")' \
	-t 'select_outputs(outputs: ["outputs"])' \
	-t 'pulse(symbol: Some("AUDIO_SIGNAL__TIME"), pulse: "112")' \
	run \
	--input-from-bundle $MODELS/$S3DIR/$MODEL.encoder.io.npz
