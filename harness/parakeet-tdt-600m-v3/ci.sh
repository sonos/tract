#!/bin/sh

set -ex

ROOT=$(realpath $(dirname $(realpath $0))/../..)
. $ROOT/.travis/ci-system-setup.sh

for rt in $TRACT_RUNTIMES
do
	gpu_assert=""
	case "$rt" in
		--cuda) gpu_assert="--assert-op-only Cuda*,Gpu*,DeviceSync*,Const,Source,STFT,Pad,IsNan,Add,Range,Cast,Eq,Div,Sub,Scan,Gather";;
		--metal) gpu_assert="--assert-op-only Metal*,Gpu*,DeviceSync*,Const,Source,STFT,Pad,IsNan,Add,Range,Cast,Eq,Div,Sub,Scan,Gather,Reduce*";;
	esac

	for m in preprocessor encoder decoder joint
	do
		# Encoder uses a patched model with upper bound assertion on S
		if [ "$m" = "encoder" ]; then
			nnef_file=nvidia--parakeet-tdt-0.6b-v3-f32f32.$m.p1.nnef.tgz
		else
			nnef_file=nvidia--parakeet-tdt-0.6b-v3-f32f32.$m.nnef.tgz
		fi
		# decoder LSTM should be inlined to a single body (no Scan left)
		# by declutter_single_loop — see #2157.
		extra_assert=""
		if [ "$m" = "decoder" ]; then
			extra_assert="--assert-op-count Scan 0"
		fi
		$CACHE_FILE \
			asr/608/nvidia--parakeet-tdt-0.6b-v3-f32f32/$nnef_file \
			asr/608/nvidia--parakeet-tdt-0.6b-v3-f32f32/nvidia--parakeet-tdt-0.6b-v3-f32f32.$m.io.npz

		$TRACT_RUN $MODELS/asr/608/nvidia--parakeet-tdt-0.6b-v3-f32f32/$nnef_file $rt \
			--nnef-tract-transformers -t transformers_detect_all run \
			--input-from-bundle $MODELS/asr/608/nvidia--parakeet-tdt-0.6b-v3-f32f32/nvidia--parakeet-tdt-0.6b-v3-f32f32.$m.io.npz \
			--assert-output-bundle $MODELS/asr/608/nvidia--parakeet-tdt-0.6b-v3-f32f32/nvidia--parakeet-tdt-0.6b-v3-f32f32.$m.io.npz \
			--approx very $gpu_assert $extra_assert
	 done
done
