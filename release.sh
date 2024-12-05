#!/bin/bash

set -e

git pull # make sure we are in sync
git push

which tomato || cargo install tomato-toml

CRATE_PATH=$1
VERSION=$2
. ./.all_crates.sh

if [ -z "$VERSION" ]
then
    echo "Usage: $0 <crate> <version>" 
    echo crates order is: $ALL_CRATES_PATH
    exit 1
fi

set -ex

if [ "$CRATE_PATH" = "all" ]
then
    for c in $ALL_CRATES_PATH
    do
        $0 $c $VERSION
    done
    exit 0
fi

crate=$(tomato get package.name $CRATE_PATH/Cargo.toml)
tomato set package.version $VERSION $CRATE_PATH/Cargo.toml
if [ "$crate" = "tract-metal" ]
then
    cargo publish -q --allow-dirty --no-verify -p $crate 
else
    cargo publish -q --allow-dirty -p $crate
fi

#./.change_crate_dep.sh $crate $VERSION
#
#cargo update

if [ "$CRATE_PATH" = "cli" ]
then
    git commit -m "release $VERSION" .
    git tag -f v"$VERSION"
    git push -f --tags
fi
