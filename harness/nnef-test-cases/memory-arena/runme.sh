#!/bin/sh

if [ `uname` = "Darwin" ] && ! ( sysctl -n machdep.cpu.brand_string | grep -q "(Virtual)" )
then

  ROOT=$(dirname $(realpath $0))/../../..
  . $ROOT/.travis/ci-system-setup.sh

  cd `dirname $0`
  set -ex

  : ${TRACT_RUN:=cargo run -p tract $CARGO_OPTS --}

  model=OpenELM-270M
  q=q40f16
  id="apple--$model-$q"
  generation=current
  nnef="llm/$generation/$id/$id.nnef.tgz"
  $CACHE_FILE $nnef

  $TRACT_RUN -v --nnef-tract-core --metal $MODELS/$nnef dump --set S=1024 --set P=0 --memory-arena found.json

  diff -u expected.json found.json
else
  echo "Skipped (memory arena test requires apple hardware)"
fi
