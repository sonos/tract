#!/bin/sh

if [ -e cargo-deny ]
then
    CARGO_DENY=`pwd`/cargo-deny
else
    CARGO_DENY="cargo deny"
fi

set -e

(cd api/rs ; $CARGO_DENY check)
(cd cli ; $CARGO_DENY check)
