#!/bin/bash
# Runs the e2e spec (examples/<example>/web/e2e.json) of every web demo in headless Chrome.
#
#   ci/web-e2e/run.sh local <dir> <out>   serves <dir>, which holds one web-demo-<example>/ per
#                                         built demo (its dist/), laid out as demo.tract.rs is
#   ci/web-e2e/run.sh live <url> <out>    checks the deployed site; E2E_COMMIT, when set, must be
#                                         the clean commit every demo's build.json names
#
# The Chrome profile in $WEB_E2E_HOME keeps the demos' weight caches between runs and starts
# over once it is a week old, so that downloading them gets tested too.

set -euo pipefail

mode=$1
src=$2
out=$3
root=$(cd "$(dirname "$0")/../.." && pwd)
home=${WEB_E2E_HOME:-$HOME/web-e2e}
mkdir -p "$home" "$out"

profile=$home/profile
if [ -d "$profile" ] && [ -n "$(find "$profile/.created" -mtime +6 2>/dev/null)" -o ! -e "$profile/.created" ]
then
    echo "starting from an empty Chrome profile"
    rm -rf "$profile"
fi
mkdir -p "$profile"
[ -e "$profile/.created" ] || touch "$profile/.created"

chrome=$("$root/ci/web-e2e/chrome.sh")

specs=()
args=()
case $mode in
    local)
        site=$(mktemp -d)
        trap 'rm -rf "$site"' EXIT
        cp "$root/examples/web-demos/index.html" "$site/"
        for spec in "$root"/examples/*/web/e2e.json
        do
            example=$(basename "$(dirname "$(dirname "$spec")")")
            path=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["path"].strip("/"))' "$spec")
            if [ -d "$src/web-demo-$example" ]
            then
                cp -R "$src/web-demo-$example" "$site/$path"
                specs+=("$spec")
            else
                echo "::error::$example has an e2e spec but was not built"
                missing=1
            fi
        done
        args=(--serve "$site" --base http://localhost:8123/)
        ;;
    live)
        specs=("$root"/examples/*/web/e2e.json)
        args=(--base "$src")
        [ -z "${E2E_COMMIT:-}" ] || args+=(--commit "$E2E_COMMIT")
        ;;
    *) echo "mode is local or live" >&2; exit 1 ;;
esac

status=0
python3 "$root/ci/web-e2e/driver.py" --chrome "$chrome" --profile "$profile" --out "$out" "${args[@]}" "${specs[@]}" || status=$?
[ -z "${GITHUB_STEP_SUMMARY:-}" ] || cat "$out/summary.md" >> "$GITHUB_STEP_SUMMARY"
[ -z "${missing:-}" ] || status=1
exit $status
