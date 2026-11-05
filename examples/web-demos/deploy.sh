#!/bin/bash
# Publishes built web demos to demo.tract.rs, purges Cloudflare's cache and checks what is served.
#
#   examples/web-demos/deploy.sh all | landing | <example>...
#
# <example> is a directory under examples/ with a web/ site; `all` is every one of them plus the
# landing page. Each demo is taken from its examples/<example>/web/dist, or from
# $DIST_ROOT/web-demo-<example> when DIST_ROOT is set (the Examples workflow's artifacts).
# A demo whose build.json is dirty is refused.
#
#   R2             rclone remote holding the tract-demo bucket (RCLONE_CONFIG locates its config)
#   CF_API_TOKEN   Cloudflare token with Cache Purge on the zone
#   CF_ZONE_ID     the zone serving demo.tract.rs
#
# The edge keeps every file until purged, and browsers revalidate each one (Cache-Control: no-cache),
# so the purge after the upload makes the whole new site visible at once.

set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
site=https://demo.tract.rs
bucket=tract-demo

[ $# -gt 0 ] || { sed -n '2,10p' "$0" | cut -c3- >&2; exit 1; }

: "${R2:?set R2 to the rclone remote of the tract-demo bucket}"
: "${CF_API_TOKEN:?set CF_API_TOKEN}"
: "${CF_ZONE_ID:?set CF_ZONE_ID}"

all_examples() {
    for spec in "$root"/examples/*/web/e2e.json
    do
        basename "$(dirname "$(dirname "$spec")")"
    done
}

landing=false
examples=()
for arg in "$@"
do
    case $arg in
        all) landing=true; examples+=($(all_examples)) ;;
        landing) landing=true ;;
        *)
            [ -f "$root/examples/$arg/web/e2e.json" ] || { echo "no web demo in examples/$arg" >&2; exit 1; }
            examples+=("$arg")
            ;;
    esac
done

dists=()
prefixes=()
for example in ${examples[@]+"${examples[@]}"}
do
    if [ -n "${DIST_ROOT:-}" ]
    then
        dist=$DIST_ROOT/web-demo-$example
    else
        dist=$root/examples/$example/web/dist
    fi
    [ -f "$dist/build.json" ] || { echo "$example: no build in $dist" >&2; exit 1; }
    read -r commit dirty < <(python3 -c 'import json, sys; b = json.load(open(sys.argv[1])); print(b["commit"], str(b["dirty"]).lower())' "$dist/build.json")
    [ "$dirty" = false ] || { echo "$example: built from a dirty tree, refusing" >&2; exit 1; }
    prefix=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["path"].strip("/"))' "$root/examples/$example/web/e2e.json")
    echo "$example -> /$prefix/ (${commit:0:9})"
    dists+=("$dist")
    prefixes+=("$prefix")
done

upload=(--checksum --ignore-times --s3-no-check-bucket --header-upload "Cache-Control: no-cache")

for i in ${!dists[@]}
do
    rclone copy "${dists[$i]}" "$R2:$bucket/${prefixes[$i]}" "${upload[@]}"
done
if $landing
then
    rclone copyto "$root/examples/web-demos/index.html" "$R2:$bucket/index.html" "${upload[@]}"
fi

purge=$(curl -sS -X POST "https://api.cloudflare.com/client/v4/zones/$CF_ZONE_ID/purge_cache" \
    -H "Authorization: Bearer $CF_API_TOKEN" -H "Content-Type: application/json" \
    --data '{"hosts":["demo.tract.rs"]}')
python3 -c 'import json, sys; d = json.loads(sys.argv[1]); sys.exit(0 if d["success"] else "purge failed: %s" % d["errors"])' "$purge"
echo "purged demo.tract.rs"

served=$(mktemp)
trap 'rm -f "$served"' EXIT
failed=0

check() {
    local url=$1 file=$2 headers
    if ! headers=$(curl -fsS -D - -o "$served" "$url" 2>&1)
    then
        echo "FAIL $url: $headers" >&2
    elif ! cmp -s "$served" "$file"
    then
        echo "FAIL $url: served file differs from $file" >&2
    elif ! echo "$headers" | grep -qi '^cache-control: no-cache'
    then
        echo "FAIL $url: served without Cache-Control: no-cache" >&2
    else
        return 0
    fi
    failed=1
}

for i in ${!dists[@]}
do
    while IFS= read -r -d '' file
    do
        check "$site/${prefixes[$i]}/${file#"${dists[$i]}"/}" "$file"
    done < <(find "${dists[$i]}" -type f -print0)
done
! $landing || check "$site/" "$root/examples/web-demos/index.html"

[ $failed = 0 ] || exit 1
echo "every uploaded file is served as built"
