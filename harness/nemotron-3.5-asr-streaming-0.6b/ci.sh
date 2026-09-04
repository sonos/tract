#!/bin/sh

set -ex

ROOT=$(realpath $(dirname $(realpath $0))/../..)
. $ROOT/.travis/ci-system-setup.sh

MODEL=nvidia--nemotron-3.5-asr-streaming-0.6b-f32f32
S3DIR=asr/634/$MODEL

for rt in $TRACT_RUNTIMES
do
	gpu_assert=""
	case "$rt" in
		--cuda) gpu_assert="--assert-op-only Cuda*,Gpu*,DeviceSync*,Const,Source,Pad,Add,Range,Cast,Eq,Div,Sub,Not";;
		--metal) gpu_assert="--assert-op-only Metal*,Gpu*,DeviceSync*,Const,Source,Pad,Add,Range,Cast,Eq,Div,Sub,Not";;
	esac

	for m in preprocessor encoder decoder joint
	do
		nnef_file=$MODEL.$m.nnef.tgz
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

# Check that the preprocessor can be pulsified.
# Pulse size (1600 audio samples, ~100ms) matches the streaming ASR example's
# PREPROC_PULSE, not the EN model's 4800 -- the two exports use different
# preprocessor chunking.
$TRACT_RUN $model_prefix.preprocessor.nnef.tgz \
	-t 'set_symbols(values: {"BATCH": 1})' \
	-t 'patch(body: "length = tract_core_shape_of(input_signal)[1];")' \
	-t 'select_inputs(inputs: ["input_signal"])' \
	-t 'select_outputs(outputs: ["processed_signal"])' \
	-t 'pulse(symbol: Some("INPUT_SIGNAL__TIME"), pulse: "1600")' \
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
			pp_assert="--assert-op-only Cuda*,Gpu*,DeviceSync*,Const,Source,Pad,PulsedSameAxisConcat,OptMulByScalar,OptSubUnicast"
			enc_assert="--assert-op-only Cuda*,Gpu*,DeviceSync*,Const,Source,PulsedRange,Not"
			;;
		--metal)
			pp_assert="--assert-op-only Metal*,Gpu*,DeviceSync*,Const,Source,Pad,PulsedSameAxisConcat,OptMulByScalar,OptSubUnicast"
			enc_assert="--assert-op-only Metal*,Gpu*,DeviceSync*,Const,Source,PulsedRange,Not"
			;;
		*) continue;;
	esac
	$TRACT_RUN $model_prefix.preprocessor.nnef.tgz $rt \
		-t 'set_symbols(values: {"BATCH": 1})' \
		-t 'patch(body: "length = tract_core_shape_of(input_signal)[1];")' \
		-t 'select_outputs(outputs: ["processed_signal"])' \
		-t 'pulse(symbol: Some("INPUT_SIGNAL__TIME"), pulse: "1600")' \
		dump -q $pp_assert
	$TRACT_RUN $model_prefix.encoder.nnef.tgz $rt \
		--nnef-tract-transformers \
		-t 'set_symbols(values: {"BATCH": 1})' \
		-t 'patch(body: "length = tract_core_shape_of(audio_signal)[2];")' \
		-t 'select_inputs(inputs: ["audio_signal", "lang_id"])' \
		-t 'select_outputs(outputs: ["outputs"])' \
		-t 'pulse(symbol: Some("AUDIO_SIGNAL__TIME"), pulse: "32")' \
		dump -q $enc_assert
done

# Check that the encoder can be pulsified.
# The encoder subsamples by 8x (three stride-2 convolutions) before the transformer.
# The chunk-window mask has P=4 transformer tokens per chunk (one attention chunk),
# so the input pulse must be 4 * 8 = 32 audio frames.
$TRACT_RUN $model_prefix.encoder.nnef.tgz \
	--nnef-tract-transformers \
	-t 'set_symbols(values: {"BATCH": 1})' \
	-t 'patch(body: "length = tract_core_shape_of(audio_signal)[2];")' \
	-t 'select_inputs(inputs: ["audio_signal", "lang_id"])' \
	-t 'select_outputs(outputs: ["outputs"])' \
	-t 'pulse(symbol: Some("AUDIO_SIGNAL__TIME"), pulse: "32")' \
	dump -q

# Check that pulsified encoder output matches batch output.
# --drop-partial-pulse truncates the input to a multiple of the pulse size,
# and the output comparison is trimmed accordingly.
$TRACT_RUN $model_prefix.encoder.nnef.tgz \
	--nnef-tract-transformers \
	-t 'set_symbols(values: {"BATCH": 1})' \
	-t 'patch(body: "length = tract_core_shape_of(audio_signal)[2];")' \
	-t 'select_inputs(inputs: ["audio_signal", "lang_id"])' \
	-t 'select_outputs(outputs: ["outputs"])' \
	-t 'pulse(symbol: Some("AUDIO_SIGNAL__TIME"), pulse: "32")' \
	run \
	--input-from-bundle $MODELS/$S3DIR/$MODEL.encoder.io.npz \
	--assert-output-bundle $MODELS/$S3DIR/$MODEL.encoder.io.npz \
	--approx very \
	--drop-partial-pulse

# The batch axis is the lane axis, so the autobatched form of the encoder keeps
# BATCH symbolic: no set_symbols, and a shape-generic patch body, since `length`
# as a scalar reshapes to [BATCH] and only typechecks at BATCH=1.
batched_patch='patch(body: "length = tract_core_cast(squeeze(sum_reduce(audio_signal, axes=[1,2]), axes=[1,2]) * 0.0, to = \"i64\") + tract_core_cast(tract_core_shape_of(audio_signal)[2], to = \"i64\");")'

$TRACT_RUN $model_prefix.encoder.nnef.tgz \
	--nnef-tract-transformers \
	-t "$batched_patch" \
	-t 'select_inputs(inputs: ["audio_signal", "lang_id"])' \
	-t 'select_outputs(outputs: ["outputs"])' \
	-t 'batchify_data_free(symbol: Some("BATCH"))' \
	-t 'pulse(symbol: Some("AUDIO_SIGNAL__TIME"), pulse: "32")' \
	dump -q \
	--assert-output-fact BATCH,1024,4,f32

# Four streams on four lanes of one state, seated wherever the worker finds them
# queued, each against the same stream run alone. The linger widens the turns
# whatever the box's scheduling, so the batch axis and the seating are exercised
# rather than the model being served one stream at a time.
TRACT_TURN_LINGER_US=400000 $TRACT_RUN $model_prefix.encoder.nnef.tgz \
	--nnef-tract-transformers \
	-t "$batched_patch" \
	-t 'select_inputs(inputs: ["audio_signal", "lang_id"])' \
	-t 'select_outputs(outputs: ["outputs"])' \
	-t 'batchify_data_free(symbol: Some("BATCH"))' \
	-t 'pulse(symbol: Some("AUDIO_SIGNAL__TIME"), pulse: "32")' \
	--autobatch-sessions 4 --hint BATCH=4 \
	run --streams 4 --turns 3 --assert-occupancy 2.5 \
	--input-from-bundle $MODELS/$S3DIR/$MODEL.encoder.io.npz \
	--approx exact \
	--drop-partial-pulse
