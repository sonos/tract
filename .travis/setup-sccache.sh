#!/bin/sh

set -ex

export SCCACHE_DIR=$HOME/.cache/sccache
export SCCACHE_CACHE_SIZE=2G

if [ -n "$GITHUB_ENV" ]
then
    echo "SCCACHE_DIR=$HOME/.cache/sccache" >> $GITHUB_ENV
    echo "SCCACHE_CACHE_SIZE=2G" >> $GITHUB_ENV
    echo "RUSTC_WRAPPER=sccache" >> $GITHUB_ENV
    echo "$HOME/.local/bin" >> $GITHUB_PATH
fi

LINK=https://github.com/mozilla/sccache/releases/download
SCCACHE_VERSION=v0.5.4

echo $HOME
if [ `uname` = "Linux" ]
then
  SCCACHE_FILE=sccache-$SCCACHE_VERSION-x86_64-unknown-linux-musl
else
  SCCACHE_FILE=sccache-$SCCACHE_VERSION-x86_64-apple-darwin
fi

mkdir -p $SCCACHE_DIR
mkdir -p $HOME/.local/bin
curl -L "$LINK/$SCCACHE_VERSION/$SCCACHE_FILE.tar.gz" | tar xz
mv -f $SCCACHE_FILE/sccache $HOME/.local/bin/sccache
chmod +x $HOME/.local/bin/sccache



