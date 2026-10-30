#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

DEPLOY='--nnef-tract-core . --runtime deploy --set T=5'

# Pooled from the config: a call goes through a worker's state and gets what
# the model gets alone, from one stream and from several hammering two workers.
TRACT_CONFIG=tract.toml $TRACT_RUN $DEPLOY run \
    --input-from-bundle io.npz --assert-output-bundle io.npz
TRACT_CONFIG=tract.toml $TRACT_RUN $DEPLOY bench --streams 8 \
    --input-from-bundle io.npz --max-time 200

# One worker and no room to wait: with eight streams calling at once, some
# call finds the worker busy and fails as overloaded.
if TRACT_CONFIG=tract.toml TRACT_POOLED__POOL_WORKERS=1 TRACT_POOLED__POOL_QUEUE=0 \
    $TRACT_RUN $DEPLOY bench --streams 8 --input-from-bundle io.npz --max-time 200 2> overloaded.log; then
    exit 1
fi
grep -q Overloaded overloaded.log
rm overloaded.log

# Pooled and autobatched at once is refused, and so is a pulsed model, which
# keeps its stream between calls.
if TRACT_CONFIG=lanes.toml $TRACT_RUN $DEPLOY dump -q; then
    exit 1
fi
if TRACT_CONFIG=tract.toml $TRACT_RUN --nnef-tract-core . -t 'pulse(symbol:Some("T"),pulse:"1")' \
    --runtime deploy dump -q; then
    exit 1
fi
