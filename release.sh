#!/bin/bash

# From: https://gist.github.com/sj26/88e1c6584397bb7c13bd11108a579746

# Retry a command up to a specific numer of times until it exits successfully,
# with exponential back off.
#
#  $ retry 5 echo Hello
#  Hello
#
#  $ retry 5 false
#  Retry 1/5 exited 1, retrying in 1 seconds...
#  Retry 2/5 exited 1, retrying in 2 seconds...
#  Retry 3/5 exited 1, retrying in 4 seconds...
#  Retry 4/5 exited 1, retrying in 8 seconds...
#  Retry 5/5 exited 1, no more retries left.
#  
function retry {
  local retries=$1
  shift

  local count=0
  until "$@"; do
    exit=$?
    wait=$((2 ** $count))
    count=$(($count + 1))
    if [ $count -lt $retries ]; then
      echo "Retry $count/$retries exited $exit, retrying in $wait seconds..."
      sleep $wait
    else
      echo "Retry $count/$retries exited $exit, no more retries left."
      return $exit
    fi
  done
  return 0
}

CRATE=$1
VERSION=$2
CRATES="data linalg core nnef pulse-opl pulse hir tensorflow onnx-opl onnx kaldi cli"

if [ `uname` = "Darwin" ]
then
    SED=gsed
else
    SED=sed
fi

if [ -z "$VERSION" ]
then
    echo "Usage: $0 <crate> <version>" 
    echo crates order is: $CRATES
    exit 1
fi

set -ex

if [ "$CRATE" = "all" ]
then
    for c in $CRATES
    do
        retry 10 $0 $c $VERSION
    done
    exit 0
fi

# set_version cargo-dinghy/Cargo.toml 0.3.0
set_version() {
    FILE=$1
    VERSION=$2
    $SED -i.back "0,/^version/s/^version *= *\".*\"/version = \"$2\"/" $FILE
    for dep in `grep "^tract-" $FILE | cut -d " " -f 1`
    do
        cargo add --manifest-path $FILE $dep@=$VERSION
    done
}

for crate in $CRATES
do
    cargo update -p tract-$crate || true
done

set_version $CRATE/Cargo.toml $VERSION
(cd $CRATE ; cargo publish --allow-dirty)

if [ "$CRATE" = "cli" ]
then
    git commit -m "release $VERSION" .
    git tag -f v"$VERSION"
    git push -f --tags
fi

cargo update
