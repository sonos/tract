# tract-distributed (Dis-tract)

Pipeline / layer-split distributed inference for [tract](https://github.com/sonos/tract):
run a model **too big for one machine** by splitting it across workers, each holding a
contiguous slice of layers plus that slice's weights and KV cache. Only the residual
activation crosses the wire — never the KV cache. Each worker may run a **different
backend** (CPU / Metal / CUDA): a shard is backend-neutral until the worker `prepare()`s
it for its own `Runtime`.

Detection and transport use **[Eclipse Zenoh](https://zenoh.io)** — the stack EXO uses:
scouting for discovery, pub/sub for the activation hot path, a queryable for assignment.

**Status: proof of concept, not proposed for merge.** See *Where this stands* below.

## The measured result

A 2-stage split runs at **whole-model speed**. Qwen3-8B-q40ef16, M-series, Metal,
`distract-shardbench` (one process, so the only variable is how the model was built):

| config | ms/step | tok/s |
|---|---|---|
| whole model (`load_model`) | 43.1 | 23.2 |
| full-range shard, **unsplit** | 42.4 | 23.6 |
| **2-shard chain** | 43.2 | 23.2 |

Splitting costs ~1.4%, token-identical. Live 2-worker cluster over zenoh: **21.4 tok/s**,
TTFT 181 ms, each node holding 2.2 GB of a 4.29 GB model.

## Run it

Models are the q40ef16 NNEF exports tract CI publishes (`Qwen3-8B-q40ef16.nnef.tgz` and
friends). The coordinator is the zenoh **router**, so start it first; workers and the
dashboard are clients that retry until it is up.

```sh
MODEL=/path/to/Qwen3-8B-q40ef16.nnef.tgz

# coordinator: plans a memory-weighted split from the workers' advertised budgets,
# serves each its shard spec, then runs as a persistent generate server.
cargo run --release -p tract-distributed --bin distract-llm -- --model "$MODEL" --workers 2

# two workers — each loads ONLY its own layers' weights (never the full model)
cargo run --release -p tract-distributed --bin distract-worker -- \
    --name node-a --backend metal --mem-mb 4096
cargo run --release -p tract-distributed --bin distract-worker -- \
    --name node-b --backend metal --mem-mb 4096

# optional: live cluster view + chat box on http://127.0.0.1:8088
DISTRACT_TOKENIZER=/path/to/tokenizer.json \
    cargo run --release -p tract-distributed --bin distract-dashboard
```

Then either chat in the dashboard, or drive it headless with token ids:

```sh
# prompt ids, max_tokens, optional stop ids
cargo run --release -p tract-distributed --bin distract-gen -- "9707,11,1246,525,498" 32
```

`--backend cpu` works the same way; a heterogeneous `cpu` + `metal` cluster is the
original demo. On a real LAN the workers find the coordinator by scouting; on loopback
they bootstrap from `tcp/127.0.0.1:7447` (macOS loopback multicast is unreliable).

## Verification

```sh
cargo test -p tract-distributed --test mlp_pipeline                    # partition + wire, bit-exact
DISTRACT_MODEL=$MODEL cargo test -p tract-distributed --test llm_pipeline -- --ignored
```

Diagnostics: `distract-shardbench <model> [backend]` (whole vs shard vs split, the table
above), `distract-metalaudit <model> <n_layers>` (device-vs-host op placement, catches
silent CPU fallback), `distract-probe` / `distract-shard-run` (per-shard parity).

## Where this stands

Honest limits, worst first:

- **Not on the public API.** The crate reaches into `tract_core`/`tract_nnef` (~38 sites)
  rather than `api/rs`, so it does not yet meet the bar `causal_llm` sets. The blocker is
  real, not laziness — see the design question below.
- **The coordinator still materialises the full model** to compute the layer weight
  profile, then drops it. So "too big for one node" is true of the *workers* but not yet
  of the cluster: the coordinator needs a box that fits the model. The profile could be
  read from the graph AST instead, as the shard loader already does.
- **`shard_graph.rs` hardcodes torch2nnef naming** (`model_model__{N}_inputLayernorm_…`).
  It works for the published q40ef16 Qwen/Llama exports; it is not a general mechanism.
- **`api/rs` needs an escape hatch.** This branch adds `Model::typed_ref` /
  `into_typed_model` (`#[doc(hidden)]`) because a `Model` cannot otherwise be split or
  measured. Shape it however you prefer — it is the smallest thing that unblocks this.
- **CPU needs #2477** (block-quant `AddUnicast` fusion guard) or Qwen decodes NaN. Metal
  is fine on current main (#2476, #2472 merged).
- Greedy decode only, one prompt at a time, KV reset per prompt (no multi-turn memory).
- CUDA compiles and self-reports via `Runtime::check()`, but is unexercised on Apple HW.

## The design question

Splitting as a registered `ModelTransform` — the shape suggested on #2465 — conflicts with
the reason this exists. A `ModelTransform` operates on an **already-loaded `TypedModel`**,
so the full model must be materialised first, which is exactly what a
too-big-for-one-machine split must avoid. This crate instead prunes the **NNEF graph AST**
and reads only the shard's `.dat` tensors (EXO's approach), which needs loader internals
that `api/rs` does not expose.

So the upstreamable primitive is probably not the splitter but a **public partial-load
API** for NNEF (load a graph subset + only its tensors). With that, Dis-tract becomes an
ordinary `causal_llm`-style example on the public API. Guidance welcome — that decision
shapes the rewrite.
