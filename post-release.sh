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
    tomato set package.version $VERSION $path/Cargo.toml > /dev/null
    for other_cargo_toml in `find . -name Cargo.toml \!  -path "./target/*" \! -path "./issue*"`
    do
        for prefix in "" "dev-" "build-" "cfg(any(target_os = \"macos\", target_os = \"ios\"
        do
            if tomato get ${prefix}dependencies.$crate $other_cargo_toml | grep -F . > /dev/null
            then
                tomato set ${prefix}dependencies.$crate.version "=$VERSION" $other_cargo_toml > /dev/null
            fi
        done
    done
done

git commit . -m "post-release $VERSION"
git push
