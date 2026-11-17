#!/bin/sh
set -eu
cd "$(dirname "$0")"
: "${TRACT_RUN:=cargo run -p tract-cli ${CARGO_OPTS:-} --}"
export TRACT_RUN
export TRACT_RUNTIMES="${TRACT_RUNTIMES:-}"
python3 check.py
