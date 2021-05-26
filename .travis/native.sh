#!/bin/sh

set -ex

./.travis/regular-tests.sh
./.travis/cli-tests.sh
