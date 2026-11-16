#!/bin/sh
set -eu
cd "$(dirname "$0")"
: "${TRACT_RUN:=cargo run -p tract-cli ${CARGO_OPTS:-} --}"
export TRACT_RUN
python3 check.py
