# tract-asset-mirror

A LAN caching reverse-proxy for tract's CI/bench model assets, backed by the R2
origin (`https://tract-test-assets.tract.rs`). Point on-LAN boxes and the device
fleet at one of these instead of each keeping its own multi-GB copy; the origin is
hit once per asset and everything else is served at local-network speed.

Not a workspace member (it is in the root `[workspace] exclude`), so its TLS/HTTP
deps stay out of tract's shared lock and default/CI builds. Build it explicitly.

## Behaviour

- `GET /<key>` — serve from the cache dir if present, else fetch once from the
  origin, store via temp-file + atomic rename, then serve. Keys are immutable
  (generation-tagged), so cached files are never revalidated.
- Concurrent first-misses of the same key coalesce to a single origin fetch.
- `HEAD /<key>` — return `Content-Length` (from disk if cached, else proxied from
  the origin) without downloading the body, for size probes.
- Path traversal is rejected; unknown keys pass the origin's 404 through.

## Run

```sh
cargo build --release
./target/release/tract-asset-mirror \
    --listen 0.0.0.0:8080 \
    --cache-dir /srv/tract-assets \
    --origin https://tract-test-assets.tract.rs
```

Flags also read env fallbacks: `LISTEN`, `CACHE_DIR`, `ORIGIN`, `THREADS`.

## Static builds (like the tract CLI)

```sh
./build-static.sh                                       # x86_64 + aarch64 -> dist/
TARGETS=armv7-unknown-linux-musleabihf ./build-static.sh  # e.g. 32-bit Raspberry Pi / Raspbian
```

Supported targets: `x86_64-unknown-linux-musl`, `aarch64-unknown-linux-musl`,
`armv7-unknown-linux-musleabihf`. Produces fully-static musl binaries (bundled
certs, no system deps -- a static musl binary runs on old glibc userlands too).
Set `CROSS_ROOT` to where the `*-cross/` toolchains live.

## Pointing clients at it

Set the asset base URL to `http://<mirror-host>:8080/`. CI and off-LAN consumers
keep using the public origin (Cloudflare-edge-cached) directly.
