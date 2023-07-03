#!/bin/sh

if [ -e cargo-deny ]
then
    CARGO_DENY=`pwd`/cargo-deny
else
    CARGO_DENY="cargo deny"
fi

for crate in onnx pulse tensorflow
do
    echo $crate:
    (cd $crate ; $CARGO_DENY check)
done
