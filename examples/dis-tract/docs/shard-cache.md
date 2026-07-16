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

## Protocol

`ModelId` = SHA-256 of `graph.nnef`. It identifies the export, and therefore the tensor label
namespace, so a cache is scoped by it and cannot mix models.

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

1. Pull `AssignSpec` (as today) -> layer range.
2. Pull `manifest/{node_id}`.
3. Cache dir: `~/.dis-tract/cache/{model_id}/`.
4. For `graph.nnef` and each entry:
   - present, `len` matches, and SHA-256 matches -> **hit**, use it;
   - otherwise -> pull the key, verify the bytes **before** writing, write to `{label}.dat.tmp`,
     `rename` into place.
5. Build the ProtoModel from the cached graph + tensors, exactly as `load_shard` does now.

`rename` is atomic, so a worker killed mid-write leaves no half file. The hash makes that
belt-and-braces rather than load-bearing: a torn file fails verification and is re-fetched.

A restart therefore costs **~0.9 s of hashing** instead of a 1.82 GB transfer — and a node
that has never seen the model pays the transfer once (~18 s on 1 GbE, ~2 s on 10 GbE),
against a 28 s shard build that dwarfs it either way.

## Consequences worth accepting deliberately

- **The coordinator must explode the archive to a directory.** A `.tgz` is a gzip stream: no
  random access. Serving 198 tensors by re-decompressing from byte zero each time is absurd,
  and holding 4.29 GB in RAM is exactly the thing we are trying to stop doing. So the
  coordinator unpacks once to `{cache}/{model_id}/` and serves files. tract's NNEF already
  loads a directory (`Nnef::proto_model_for_path` walks it when the path is not a file), so
  this is a supported layout, not a private format.
- **The coordinator becomes the single source.** EXO pulls from HF, so its workers do not
  depend on one box's disk or uplink; ours would. A shared store or an HTTP/S3 origin is the
  obvious escape, and the manifest is agnostic about who serves the bytes.
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
