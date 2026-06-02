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
		--cuda) gpu_assert="--assert-op-only Cuda*,Gpu*,DeviceSync*,Const,Source,STFT,Pad,Add,Range,Cast,Eq,Div,Sub";;
		--metal) gpu_assert="--assert-op-only Metal*,Gpu*,DeviceSync*,Const,Source,STFT,Pad,Add,Range,Cast,Eq,Div,Sub";;
	esac

	for m in preprocessor encoder decoder joint
	do
		# Encoder uses a patched model with upper bound assertion on AUDIO_SIGNAL__TIME
		if [ "$m" = "encoder" ]; then
			nnef_file=$MODEL.$m.p1.nnef.tgz
		else
			nnef_file=$MODEL.$m.nnef.tgz
		fi
		# Decoder is stepped one token per call by the caller (external state
		# carry): assert the external_state flag and concretize the seq symbol
		# to 1 so the Scan inlines and the LSTM body lands on the GPU instead of
		# bouncing through CPU each step. set_symbols RON must stay space-free
		# ($extra_transforms is passed unquoted).
		extra_transforms=""
		if [ "$m" = "decoder" ]; then
			extra_transforms='-t force_scan_external_state -t set_symbols(values:{"TARGETS__TIME":1})'
		fi
		$CACHE_FILE \
			$S3DIR/$nnef_file \
			$S3DIR/$MODEL.$m.io.npz

		$TRACT_RUN $MODELS/$S3DIR/$nnef_file $rt --nnef-tract-transformers -t transformers_detect_all $extra_transforms run \
			--input-from-bundle $MODELS/$S3DIR/$MODEL.$m.io.npz --assert-output-bundle $MODELS/$S3DIR/$MODEL.$m.io.npz \
			--approx very $gpu_assert
	done
done

model_prefix=$MODELS/$S3DIR/$MODEL

# Check that the patch transform eliminates all Iff nodes,
# and that select_outputs can reduce the model to a single output
$TRACT_RUN $model_prefix.preprocessor.nnef.tgz \
	-t 'set_symbols(values: {"BATCH": 1})' \
	-t 'patch(body: "length = tract_core_shape_of(input_signal)[1];")' \
	-t 'select_inputs(inputs: ["input_signal"])' \
	-t 'select_outputs(outputs: ["processed_signal"])' \
	dump -q \
	--assert-op-count Iff 0

# Check that the preprocessor can be pulsified
$TRACT_RUN $model_prefix.preprocessor.nnef.tgz \
	-t 'set_symbols(values: {"BATCH": 1})' \
	-t 'patch(body: "length = tract_core_shape_of(input_signal)[1];")' \
	-t 'select_inputs(inputs: ["input_signal"])' \
	-t 'select_outputs(outputs: ["processed_signal"])' \
	-t 'pulse(symbol: Some("INPUT_SIGNAL__TIME"), pulse: "4800")' \
	dump -q

# Check that pulsified preprocessor and encoder translate cleanly on each GPU
# runtime (the GPU translator must fall back to CPU for ops it can't lower, not
# abort the whole transform).  Allowlist what currently falls back so a
# regression spilling another op to CPU fails CI.  Runtime numeric checks are
# deferred; only the translation is asserted here.
for rt in $TRACT_RUNTIMES
do
	case "$rt" in
		--cuda)
			pp_assert="--assert-op-only Cuda*,Gpu*,DeviceSync*,Const,Source,STFT,Pad,PulsedSameAxisConcat,OptMulByScalar,OptSubUnicast"
			enc_assert="--assert-op-only Cuda*,Gpu*,DeviceSync*,Const,Source,AffineChunkTrim,PulsedRange"
			;;
		--metal)
			pp_assert="--assert-op-only Metal*,Gpu*,DeviceSync*,Const,Source,STFT,Pad,PulsedSameAxisConcat,OptMulByScalar,OptSubUnicast"
			enc_assert="--assert-op-only Metal*,Gpu*,DeviceSync*,Const,Source,AffineChunkTrim,PulsedRange"
			;;
		*) continue;;
	esac
	$TRACT_RUN $model_prefix.preprocessor.nnef.tgz $rt \
		-t 'set_symbols(values: {"BATCH": 1})' \
		-t 'patch(body: "length = tract_core_shape_of(input_signal)[1];")' \
		-t 'select_outputs(outputs: ["processed_signal"])' \
		-t 'pulse(symbol: Some("INPUT_SIGNAL__TIME"), pulse: "4800")' \
		dump -q $pp_assert
	$TRACT_RUN $model_prefix.encoder.p1.nnef.tgz $rt \
		--nnef-tract-transformers \
		-t 'set_symbols(values: {"BATCH": 1})' \
		-t 'patch(body: "length = tract_core_shape_of(audio_signal)[2];")' \
		-t 'select_outputs(outputs: ["outputs"])' \
		-t 'pulse(symbol: Some("AUDIO_SIGNAL__TIME"), pulse: "112")' \
		dump -q $enc_assert
done

# Check that the encoder can be pulsified.
# The encoder subsamples by 8x (three stride-2 convolutions) before the transformer.
# The chunk-window mask has P=14 transformer tokens per chunk, so the input pulse
# must be 14 * 8 = 112 audio frames.
$TRACT_RUN $model_prefix.encoder.p1.nnef.tgz \
	--nnef-tract-transformers \
	-t 'set_symbols(values: {"BATCH": 1})' \
	-t 'patch(body: "length = tract_core_shape_of(audio_signal)[2];")' \
	-t 'select_inputs(inputs: ["audio_signal"])' \
	-t 'select_outputs(outputs: ["outputs"])' \
	-t 'pulse(symbol: Some("AUDIO_SIGNAL__TIME"), pulse: "112")' \
	dump -q

# Check that pulsified encoder output matches batch output.
# --drop-partial-pulse truncates the input to a multiple of the pulse size,
# and the output comparison is trimmed accordingly.
$TRACT_RUN $model_prefix.encoder.p1.nnef.tgz \
	--nnef-tract-transformers \
	-t 'set_symbols(values: {"BATCH": 1})' \
	-t 'patch(body: "length = tract_core_shape_of(audio_signal)[2];")' \
	-t 'select_inputs(inputs: ["audio_signal"])' \
	-t 'select_outputs(outputs: ["outputs"])' \
	-t 'pulse(symbol: Some("AUDIO_SIGNAL__TIME"), pulse: "112")' \
	run \
	--input-from-bundle $MODELS/$S3DIR/$MODEL.encoder.io.npz \
	--assert-output-bundle $MODELS/$S3DIR/$MODEL.encoder.io.npz \
	--approx very \
	--drop-partial-pulse
