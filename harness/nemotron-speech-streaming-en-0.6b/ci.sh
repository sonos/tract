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
		$CACHE_FILE \
			$S3DIR/$MODEL.$m.nnef.tgz \
			$S3DIR/$MODEL.$m.io.npz

		model_prefix=$MODELS/$S3DIR/$MODEL

		$TRACT_RUN $model_prefix.$m.nnef.tgz $rt --nnef-tract-transformers -t transformers_detect_all run \
			--input-from-bundle $model_prefix.$m.io.npz --assert-output-bundle $model_prefix.$m.io.npz \
			--approx very
	done
done
