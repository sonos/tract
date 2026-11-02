#!/bin/bash
# Prints the path of the Chrome for Testing build ci/web-e2e/chrome-for-testing.version pins,
# downloading it into $WEB_E2E_HOME/chrome/<version>/ the first time.

set -euo pipefail

home=${WEB_E2E_HOME:-$HOME/web-e2e}
version=$(cat "$(dirname "$0")/chrome-for-testing.version")
case "$(uname -s)/$(uname -m)" in
    Darwin/arm64)
        platform=mac-arm64
        binary="chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing" ;;
    Linux/x86_64)
        platform=linux64
        binary=chrome-linux64/chrome ;;
    *) echo "no Chrome for Testing build for $(uname -s)/$(uname -m)" >&2; exit 1 ;;
esac

dir=$home/chrome/$version
if [ ! -x "$dir/$binary" ]
then
    rm -rf "$dir.tmp"
    mkdir -p "$dir.tmp"
    curl -fsSL -o "$dir.tmp/chrome.zip" \
        "https://storage.googleapis.com/chrome-for-testing-public/$version/$platform/chrome-$platform.zip"
    (cd "$dir.tmp" && unzip -q chrome.zip && rm chrome.zip)
    rm -rf "$dir"
    mv "$dir.tmp" "$dir"
fi
echo "$dir/$binary"
