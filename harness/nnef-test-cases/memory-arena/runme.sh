#!/bin/sh

ROOT=$(dirname $(realpath $0))/../../..
cd `dirname $0`

if [ `uname` = "Darwin" ] && ! ( sysctl -n machdep.cpu.brand_string | grep -q "(Virtual)" )
then
  DEVICE=--metal
  expected=expected.metal.json
elif [ `uname` = "Linux" ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
then
  DEVICE=--cuda
  expected=expected.cuda.json
else
  echo "Skipped (memory arena requires Apple Metal or an NVIDIA GPU)"
  exit 0
fi

. $ROOT/.travis/ci-system-setup.sh

set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

id="apple--OpenELM-270M-q40ef16"
generation=541
nnef="llm/$generation/$id/$id.nnef.tgz"
$CACHE_FILE $nnef

$TRACT_RUN -v --nnef-tract-transformers $DEVICE $MODELS/$nnef dump --set S=1024 --set P=0 --memory-arena found.json

if [ -n "$RESET" ]
then
  cp found.json $expected
else
  diff -u $expected found.json
fi
