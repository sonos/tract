#!/bin/sh

set -ex

ROOT=$(realpath $(dirname $(realpath $0))/../..)
. $ROOT/.travis/ci-system-setup.sh

for rt in $TRACT_RUNTIMES
do
	for m in preprocessor encoder decoder joint
	do
		$CACHE_FILE \
		asr/608/nvidia--parakeet-tdt-0.6b-v3-f32f32/nvidia--parakeet-tdt-0.6b-v3-f32f32.$m.nnef.tgz \
	 	asr/608/nvidia--parakeet-tdt-0.6b-v3-f32f32/nvidia--parakeet-tdt-0.6b-v3-f32f32.$m.io.npz

	 	model_prefix=$MODELS/asr/608/nvidia--parakeet-tdt-0.6b-v3-f32f32/nvidia--parakeet-tdt-0.6b-v3-f32f32

	 	$TRACT_RUN $model_prefix.$m.nnef.tgz $rt --nnef-tract-transformers -t transformers-detect-all run \
	 		--input-from-bundle $model_prefix.$m.io.npz --assert-output-bundle $model_prefix.$m.io.npz \
	 		--approx very
	 done
done
