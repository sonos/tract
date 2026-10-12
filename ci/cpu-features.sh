#!/bin/sh
# Which silicon this job drew. The hosted-runner fleet is mixed -- AMD parts without avx512 next
# to Intel parts with amx -- so a green run says nothing about whether the kernels behind a
# feature ran at all, unless the job says what it had.

if [ "$(uname)" = "Darwin" ]
then
    sysctl -n machdep.cpu.brand_string
    exit 0
fi

[ -e /proc/cpuinfo ] || exit 0

grep -m1 -E '^(model name|Model)' /proc/cpuinfo || true
grep -m1 -E '^(flags|Features)' /proc/cpuinfo | cut -d: -f2 | tr ' ' '\n' \
    | grep -xE 'avx|avx2|fma|f16c|avx_vnni|avx512[a-z0-9_]*|amx_[a-z0-9]+|neon|asimdhp|asimddp|i8mm|bf16|sve|sve2|sme|sme2' \
    | sort | tr '\n' ' ' || true
echo
