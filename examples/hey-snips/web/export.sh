#!/bin/sh
# The graph tar the demo loads, from the hey_snips v4 TensorFlow model in tract's test assets:
#
#   ./export.sh <out dir>
#
# tract dumps the wavenet as NNEF, export.py puts the resampler and featurizer in front of it, and
# tract pulses the whole graph by 3840 samples at 48 kHz (8 frames of 10 ms).

set -e
OUT=`realpath -m $1`
cd `dirname $0`
: ${TRACT_RUN:=cargo run -q --release -p tract-cli --}
: ${TRACT_MODELS_URL:=https://tract-test-assets.tract.rs}

WORK=`mktemp -d`
trap "rm -rf $WORK" EXIT
curl -fsSL $TRACT_MODELS_URL/hey_snips_v4_model17.pb -o $WORK/hey_snips_v4_model17.pb
$TRACT_RUN $WORK/hey_snips_v4_model17.pb -i S,20,f32 dump -q --nnef-dir $WORK/wavenet
python3 export.py $WORK/wavenet $WORK/graph
mkdir -p $OUT
$TRACT_RUN $WORK/graph --nnef-tract-core --pulse 3840 --nnef-tract-pulse dump -q --nnef-tar $OUT/hey_snips.nnef.tar
ls -l $OUT
