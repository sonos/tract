#!/bin/sh

if [ -e cargo-deny ]
then
    CARGO_DENY=`pwd`/cargo-deny
else
    CARGO_DENY="cargo deny"
fi

(cd api/rs ; $CARGO_DENY check)
