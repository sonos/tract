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
    echo $crate
    tomato set package.version $VERSION $path/Cargo.toml > /dev/null
    tomato set workspace.dependencies.$crate.version $VERSION Cargo.toml
done

git commit . -m "post-release $VERSION"
git push
