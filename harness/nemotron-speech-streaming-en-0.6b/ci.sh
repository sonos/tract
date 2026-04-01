#!/bin/sh

set -ex

ROOT=$(realpath $(dirname $(realpath $0))/../..)
. $ROOT/.travis/ci-system-setup.sh

MODEL=nvidia--nemotron-speech-streaming-en-0.6b-f32f32
S3DIR=asr/613/$MODEL

for rt in $TRACT_RUNTIMES
do
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
			--approx very
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

# Check that the preprocessor can be pulsified
$TRACT_RUN $model_prefix.preprocessor.nnef.tgz \
	-t 'concretize_symbols(values: {"BATCH": 1})' \
	-t 'patch(body: "length = tract_core_shape_of(input_signal)[1];")' \
	-t 'select_outputs(outputs: ["processed_signal"])' \
	-t 'pulse(symbol: Some("INPUT_SIGNAL__TIME"), pulse: "4800")' \
	dump -q
