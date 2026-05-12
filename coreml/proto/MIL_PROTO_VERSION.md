# Vendored MIL / Model proto files

These `.proto` files come from `apple/coremltools` and are processed by `prost-build` in `build.rs` to generate Rust types.

| Field | Value |
|---|---|
| Source repo | https://github.com/apple/coremltools |
| Pinned tag | `9.0` |
| Pinned commit | `428d4b2658dfc44194f27f4f36870751be402ff7` |
| Path in repo | `mlmodel/format/*.proto` |
| Files | 33 (~266 KB raw) |
| Vendored | 2026-05-09 |

## Re-vendor

```sh
SHA=428d4b2658dfc44194f27f4f36870751be402ff7
DEST=~/coding/tract-coreml-ane/coreml-hello/proto
rm -rf $DEST && mkdir -p $DEST
# Then iterate the contents API for the format/ directory at $SHA and curl each .proto
```

## License

All files are `BSD-3-Clause` per `apple/coremltools/LICENSE.txt`. No header changes; we treat them as unmodified third-party.
