#!/bin/sh

set -ex

ROOT=$(realpath $(dirname $(realpath $0))/../..)
. $ROOT/.travis/ci-system-setup.sh

for rt in $TRACT_RUNTIMES
do
	for m in preprocessor encoder decoder joint
	do
		# Encoder uses a patched model with upper bound assertion on S
		if [ "$m" = "encoder" ]; then
			nnef_file=nvidia--parakeet-tdt-0.6b-v3-f32f32.$m.p1.nnef.tgz
		else
			nnef_file=nvidia--parakeet-tdt-0.6b-v3-f32f32.$m.nnef.tgz
		fi
		$CACHE_FILE \
			asr/608/nvidia--parakeet-tdt-0.6b-v3-f32f32/$nnef_file \
			asr/608/nvidia--parakeet-tdt-0.6b-v3-f32f32/nvidia--parakeet-tdt-0.6b-v3-f32f32.$m.io.npz

		$TRACT_RUN $MODELS/asr/608/nvidia--parakeet-tdt-0.6b-v3-f32f32/$nnef_file $rt \
			--nnef-tract-transformers -t transformers_detect_all run \
			--input-from-bundle $MODELS/asr/608/nvidia--parakeet-tdt-0.6b-v3-f32f32/nvidia--parakeet-tdt-0.6b-v3-f32f32.$m.io.npz \
			--assert-output-bundle $MODELS/asr/608/nvidia--parakeet-tdt-0.6b-v3-f32f32/nvidia--parakeet-tdt-0.6b-v3-f32f32.$m.io.npz \
			--approx very
	 done
done
