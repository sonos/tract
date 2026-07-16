# Shard distribution and the worker cache (design, not implemented)

## The problem

`AssignSpec` sends a **path**, so today every worker needs the whole model file at that same
path on its own disk, and streams/decompresses all of it (4.29 GB for Qwen3-8B) to extract
the ~1.8 GB it owns. Two consequences:

- A bare node cannot join. Getting the model to each node is out of scope — scp/NFS/rsync it
  yourself, and `--model` must resolve identically everywhere.
- A model too big for a node's **disk** cannot run at all, even though the point is only that
  it is too big for its **RAM**.

EXO does not have this problem: it reads the safetensors index, works out which files hold
its layers, and range-downloads only those. A node running layers 0-17 never has 18-35 on
disk. We want that property, with the coordinator as the store.

## Why the old objection does not apply

Shipping shards was abandoned because `to_proto_model(&TypedModel)` fails on Qwen's EinSum,
which blocked serializing a sub-model to NNEF. That serializer is not needed here.

`load_shard` never round-trips a `TypedModel`. It parses graph text, prunes the AST, and
reads raw `.dat` bytes. Those two artefacts are exactly what must cross the wire, and both
are already in the form we would send:

- `graph.nnef` is **0.91 MB** — send it verbatim, no pruning or printing on the coordinator.
  The worker prunes locally, as it already does.
- A `.dat` file is a self-describing tensor (`tract_nnef::tensors::read_tensor` parses it).
  Send the bytes untouched.

Nothing is re-serialized. The failing path is never entered.

## Sizes that shape the design

| | |
|---|---|
| graph.nnef | 0.91 MB |
| all 545 tensors | 4.29 GB |
| one shard (layers 18-35) | 1.82 GB across 198 tensors |
| tensor size | min 0.1 KB, median 0.01 MB, **max 333.8 MB** (embedding) |

The median tensor is tiny and the max is enormous: chunk per tensor, and the embedding still
needs range/streaming rather than one payload.

## Integrity: SHA-256, not MD5

Measured on this box (Apple silicon, 200 MB of real model bytes, extrapolated to 1.82 GB):

| algorithm | throughput | verify a shard |
|---|---|---|
| MD5 | 560 MB/s | 3.3 s |
| BLAKE2b | 941 MB/s | 2.0 s |
| **SHA-256** | **1999 MB/s** | **0.9 s** |

SHA-256 wins because ARMv8 crypto extensions accelerate it in hardware — the usual
"MD5 is faster" intuition is inverted here. It is also already in the dependency tree
(`sha2`, via `pest_meta`), so it costs no new compile. MD5 would be 3.7x slower, weaker, and
no cheaper.

Cost matters because verification runs on **every** restart, not just on fetch.

## Sources: a shared path and the wire, not one or the other

A single-source design is wrong for both ends of the range this has to serve:

- **One LAN with a NAS.** The operator already has the model on shared storage. Shipping
  1.8 GB per worker over zenoh when the bytes are one mount away is pure waste, and it makes
  the coordinator's uplink the bottleneck for no reason.
- **Nodes across sites.** No shared mount exists, and there is nothing to point a path at.

Once every tensor carries a SHA-256, **the source stops mattering**: anything that yields
bytes matching the manifest is valid. So sources become a resolution order, not a mode, and
the same verification covers all of them — including a stale or half-written file on the NAS,
which a bare path today would load silently.

Per tensor, the worker resolves:

1. **local cache** — `{cache}/{model_id}/{label}.dat`, verified;
2. **shared path**, if `model_path` is set and resolves on this node;
3. **the wire** — `distract/weights/{model_id}/{label}` from the coordinator.

Each is hash-checked, so a miss, a corrupt NAS copy, and an absent mount all degrade the
same way: fall through to the next source. A node with the mount never touches the wire; a
remote node never needs one; a mixed cluster needs no special configuration, because each
worker resolves for itself.

```rust
/// Where a worker may find its tensors, in order of preference. Both are optional:
/// a path-only cluster never serves bytes, a serve-only cluster needs nothing on
/// the worker's disk, and a mixed one lets each node use whatever it has.
pub struct ShardSource {
    /// Resolvable on the worker: an NNEF **directory** (per-file reads, so a mount
    /// serves only what each shard needs) or a `.tgz` (streamed whole — see below).
    pub model_path: Option<String>,
    /// Whether the coordinator will serve tensors for anything not found locally.
    pub serve: bool,
}
```

**On a NAS, explode the archive.** A `.tgz` is a gzip stream: a worker must pull all 4.29 GB
across NFS to reach the 1.8 GB it owns — the worst case of every option here, and worse than
fetching over zenoh. An NNEF **directory** on the NAS lets each worker read only its own
`.dat` files, which is the whole point of a shared mount. tract loads a directory natively
(`Nnef::proto_model_for_path` walks it when the path is not a file), so this is a supported
layout, and it is the same unpacking the coordinator needs in order to serve.

Whether a worker caches what it read from a path is policy: on a fast NAS the mount *is* the
cache and copying wastes disk; on a slow or contended one, caching makes restarts local.
Default to caching, with `--no-cache-from-path` to opt out.

## Protocol

`ModelId` = SHA-256 of `graph.nnef`. It identifies the export, and therefore the tensor label
namespace, so a cache is scoped by it and cannot mix models. It also lets a worker check that
the path it was handed holds the model the coordinator planned against, rather than trusting
that a mount is what it claims to be.

New keys, all served by the coordinator:

```
distract/graph/{model_id}            -> graph.nnef bytes (0.91 MB)
distract/manifest/{node_id}          -> ShardManifest for that node's assigned range
distract/weights/{model_id}/{label}  -> raw .dat bytes for one tensor
```

Weights are keyed by `model_id`, not `node_id`: co-located workers share a cache, and a node
re-assigned a different layer range reuses whatever overlaps.

```rust
/// One tensor the shard needs, and what it must hash to.
pub struct TensorEntry {
    pub label: String,   // "model.model.layers.28.mlp.up_proj.weight"
    pub len: u64,
    pub sha256: String,  // hex
}

/// Exactly the tensors a node's assigned layer range requires. `plan_shard`
/// already computes this set (`weight_labels`).
pub struct ShardManifest {
    pub model_id: String,
    pub graph_sha256: String,
    pub entries: Vec<TensorEntry>,
}
```

## Worker flow

1. Pull `AssignSpec` (as today) -> layer range and `ShardSource`.
2. Pull `manifest/{node_id}`.
3. Cache dir: `~/.dis-tract/cache/{model_id}/`.
4. For `graph.nnef` and each entry, take the first source that yields bytes hashing to the
   manifest's value:
   - **cache** — `{label}.dat` present, `len` matches, SHA-256 matches;
   - **path** — `model_path` set and resolves: read that tensor's file (directory layout) or
     stream the archive once for all of them (`.tgz`);
   - **wire** — pull `weights/{model_id}/{label}`.
   Bytes from path or wire are verified **before** being written to `{label}.dat.tmp` and
   `rename`d into place.
5. Build the ProtoModel from the resolved graph + tensors, exactly as `load_shard` does now.

`rename` is atomic, so a worker killed mid-write leaves no half file. The hash makes that
belt-and-braces rather than load-bearing: a torn file fails verification and falls through to
the next source.

A restart therefore costs **~0.9 s of hashing** instead of any transfer at all. A node that
has never seen the model pays once — free from a mount, ~18 s over 1 GbE, ~2 s over 10 GbE —
against a 28 s shard build that dwarfs it either way. Which is the point: **the first load is
not the cost worth optimising; the repeated one is.**

Failure to resolve a tensor from *any* source is a hard error naming the label and the
sources tried. Today a missing `.dat` surfaces as `loaded N of M shard weights`, which does
not say where it looked.

## Consequences worth accepting deliberately

- **The coordinator must explode the archive to a directory** *if it serves*. A `.tgz` is a
  gzip stream: no random access. Serving 198 tensors by re-decompressing from byte zero each
  time is absurd, and holding 4.29 GB in RAM is exactly the thing we are trying to stop
  doing. So it unpacks once to `{cache}/{model_id}/` and serves files. tract's NNEF already
  loads a directory (`Nnef::proto_model_for_path` walks it when the path is not a file), so
  this is a supported layout, not a private format. A path-only cluster skips this entirely —
  the coordinator never opens the weights.
- **`serve` centralises on the coordinator; a path does not.** EXO pulls from HF, so its
  workers never depend on one box's disk or uplink. With `serve`, ours would — which is
  precisely why the path source is not a legacy option to be removed. On a LAN with shared
  storage, `model_path` pointed at a NAS directory is strictly better than serving: no
  uplink bottleneck, no coordinator disk, workers fetch only their own files. `serve` earns
  its keep exactly where a mount cannot reach — across sites — and the two coexist because
  the manifest, not the source, is what defines correctness.
- **The cache is unbounded.** Nothing evicts. `{model_id}` scoping means a new export lands
  beside the old one rather than replacing it. Needs a size cap or an LRU before it is a
  real feature.
- **Two workers on one host race** to fetch the same tensor. Atomic rename keeps the cache
  correct; the waste is duplicate transfer. A per-label lock file would fix it if it matters.
- **Verification is not free.** ~0.9 s per restart, and it should probably be skippable
  (`--trust-cache`) for a developer loop, with the default staying safe.
- **This does not fix the coordinator materialising the model.** It still loads the whole
  model to compute the layer weight profile. Until that profile is read from the AST, the
  coordinator needs a node that fits the model, which is the case we most want to serve.

## Measured: zenoh's payload ceiling

`distract-wiretest` serves N MB of non-uniform bytes from a queryable and pulls them back
through a router on its own port, checking length and a byte sum so a truncated or
zero-filled reply cannot pass. Loopback, Apple silicon:

| size | time | throughput | intact |
|---|---|---|---|
| 1 MB | 0.00 s | 293 MB/s | yes |
| 64 MB | 0.07 s | 880 MB/s | yes |
| **334 MB** (embedding) | 0.34 s | 993 MB/s | yes |
| 512 MB | 0.49 s | 1054 MB/s | yes |
| 768 MB | 0.66 s | 1170 MB/s | yes |
| 960 MB | 1.64 s | 585 MB/s | yes |
| 1024 MB | — | — | **no reply** |

**Chunk-per-tensor is viable**: the largest tensor in the model crosses in one payload,
intact. **A whole shard in one reply is not**: there is a hard wall just under 1 GiB, and
throughput already halves above 768 MB because the payload is buffered whole at both ends —
so small chunks also bound coordinator memory, not just risk.

These are loopback figures — ~1 GB/s is memcpy-bound. A 1 GbE LAN caps near 110 MB/s, so a
1.82 GB first fetch is ~18 s there whatever zenoh can do locally. What this settles is the
ceiling and integrity, which is what the design turned on.
