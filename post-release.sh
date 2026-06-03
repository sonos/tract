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

if ! echo "$VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9.-]+)?$'
then
    echo "Refusing version '$VERSION': must look like 0.23.0 or 0.23.0-pre (no leading 'v')." >&2
    exit 1
fi

for path in $ALL_CRATES_PATH
do
    crate=$(tomato get package.name $path/Cargo.toml)
    echo $crate
    tomato set package.version $VERSION $path/Cargo.toml > /dev/null
    tomato set workspace.dependencies.$crate.version $VERSION Cargo.toml
done

# tomato edits the manifests only; sync Cargo.lock to the bumped workspace
# versions so the committed lock matches (otherwise the next build rewrites it).
cargo update --workspace --offline

git commit . -m "post-release $VERSION"
git push
