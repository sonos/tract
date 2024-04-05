#!/bin/sh

VERSION=$1
. ./.all_crates.sh

if [ `uname` = "Darwin" ]
then
    SED=gsed
else
    SED=sed
fi

if [ -z "$VERSION" ]
then
    echo "Usage: $0 <version>" 
    exit 1
fi

for path in $ALL_CRATES_PATH
do
    crate=$(tomato get package.name $path/Cargo.toml)
    cargo yank --version $VERSION $crate
done
