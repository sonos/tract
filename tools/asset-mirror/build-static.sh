#!/usr/bin/env bash
#
# Build fully-static musl binaries of tract-asset-mirror. rustls + webpki-roots
# mean HTTPS to the origin works with bundled Mozilla roots -- no system libs and
# no cert store on the target box, same as the tract CLI is shipped.
#
#   ./build-static.sh                                  # x86_64 + aarch64
#   TARGETS=aarch64-unknown-linux-musl ./build-static.sh
#   CROSS_ROOT=/path/to/toolchains ./build-static.sh
#
# CROSS_ROOT holds the musl cross toolchains as <arch>-linux-musl-cross/ (default:
# the tract repo root, where CI unpacks them). Artifacts are copied to dist/<target>/.

set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
CROSS_ROOT=${CROSS_ROOT:-$(cd "$HERE/../.." && pwd)}
TARGETS=${TARGETS:-"x86_64-unknown-linux-musl aarch64-unknown-linux-musl"}

for t in $TARGETS; do
    case $t in
        x86_64-unknown-linux-musl) prefix=x86_64-linux-musl ;;
        aarch64-unknown-linux-musl) prefix=aarch64-linux-musl ;;
        armv7-unknown-linux-musleabihf) prefix=armv7l-linux-musleabihf ;;
        *) echo "unsupported target: $t" >&2; exit 1 ;;
    esac
    gcc=$CROSS_ROOT/$prefix-cross/bin/$prefix-gcc
    ar=$CROSS_ROOT/$prefix-cross/bin/$prefix-ar
    [ -x "$gcc" ] || { echo "cross gcc not found: $gcc (set CROSS_ROOT)" >&2; exit 1; }

    rustup target add "$t" >/dev/null 2>&1 || true
    up=$(printf '%s' "$t" | tr 'a-z-' 'A-Z_')
    us=${t//-/_}
    echo "==> $t  ($gcc)"
    env \
        "CARGO_TARGET_${up}_LINKER=$gcc" \
        "CC_${us}=$gcc" \
        "AR_${us}=$ar" \
        RUSTFLAGS="-C target-feature=+crt-static" \
        cargo build --release --target "$t"

    mkdir -p "$HERE/dist/$t"
    cp "$HERE/target/$t/release/tract-asset-mirror" "$HERE/dist/$t/"
    echo "    -> dist/$t/tract-asset-mirror"
done
