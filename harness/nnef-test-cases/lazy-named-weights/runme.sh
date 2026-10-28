#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# The weights live in a weights.safetensors this case does not carry: loaded lazily, they are
# never read, and the dump refers to them by name and datum type.
$TRACT_RUN model.onnx --onnx-lazy-external-data --nnef-tract-core dump -q --nnef-graph found

version=`cargo metadata --format-version 1 | jq -r '.packages | map(select( (.name) == "tract-core") | .version) | .[] '`
perl -pi -e "s/$version/0.18.3-pre/" found

diff -u expected found

# The bundle of that graph carries no weight either: it loads typed, still unread, and the
# weights dump back as the same named weights.
rm -rf bundle found-again
$TRACT_RUN model.onnx --onnx-lazy-external-data --nnef-tract-core dump -q --nnef-dir bundle
test ! -e bundle/dense.weight.dat
$TRACT_RUN bundle --nnef-tract-core dump -q --nnef-graph found-again
grep tract_core_named_weight expected | sort > weights-expected
grep tract_core_named_weight found-again | sort > weights-found
diff -u weights-expected weights-found
rm -rf bundle found found-again weights-expected weights-found
