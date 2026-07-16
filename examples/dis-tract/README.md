# tract-distributed (Dis-tract)

Pipeline / layer-split inference for [tract](https://github.com/sonos/tract): split a model
across workers, each holding a contiguous slice of layers plus that slice's weights and KV
cache. Only the residual activation crosses the wire — never the KV cache. Each worker may
run a **different backend** (CPU / Metal / CUDA): a shard is backend-neutral until the
worker `prepare()`s it for its own `Runtime`.

Transport is **[Eclipse Zenoh](https://zenoh.io)** — the stack EXO uses: scouting for
discovery, pub/sub for the activation hot path, a queryable for assignment.

**Status: proof of concept, not proposed for merge.**

The goal is to run a model too big for one machine. That is **not yet demonstrated**: every
run so far is two worker processes on a *single* machine over loopback, on a model
(Qwen3-8B-q40ef16, 4.29 GB) that fits that machine on its own. What is exercised is the
mechanism — per-shard weight loading, resident KV, the wire protocol, and token parity
against a single-machine reference. The numbers below carry no real network cost, and
zenoh's scouting is bypassed for a fixed loopback endpoint. See *Where this stands*.

## The measured result (single machine, loopback)

A 2-stage split runs at **whole-model speed**. Qwen3-8B-q40ef16, M-series, Metal,
`distract-shardbench` (one process, so the only variable is how the model was built):

| config | ms/step (min of 14) | tok/s |
|---|---|---|
| whole model (`load_model`) | 42.5 | 23.5 |
| full-range shard, **unsplit** | 42.4 | 23.6 |
| **2-shard chain** | 43.0 | 23.3 |

Token-identical; splitting costs ~1% **in-process, with no wire between the stages**. Min
rather than mean: the means carry scheduler outliers on a laptop. A live 2-worker cluster —
both workers on the same machine, zenoh over loopback — does **21-22 tok/s**, TTFT ~190 ms,
each worker holding 2.2 GB of a 4.29 GB model.

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

## More

- [docs/FAQ.md](docs/FAQ.md) — why the cluster's tok/s is half a node's, why a shard can be
  slow on GPU but not CPU, why the KV never crosses the wire, and what is *not* tested.
- [docs/upstream-partition-api.md](docs/upstream-partition-api.md) — the partition primitive.

## Where this stands

Honest limits, worst first:

- **Never run on more than one machine.** Two processes on one box over loopback, on a
  model that fits it. So the real network cost per token is unmeasured, both stages share
  one GPU, and zenoh scouting is bypassed for a fixed loopback endpoint. Two 16 GB machines
  and a model needing both (Qwen3-32B, ~17.6 GB) is the experiment that would earn the
  distributed claim.
- **Not on the public API.** The crate reaches into `tract_core`/`tract_nnef`/
  `tract_transformers` at 35 import sites rather than `api/rs`, against `causal_llm`'s
  single `use tract::`. The blocker is real, not laziness — see the design question below.
- **The coordinator still materialises the full model** to compute the layer weight
  profile, then drops it. So even on two boxes the coordinator needs one that fits the
  whole model — for the interesting case (a model too big for any single node) that is a
  hard blocker, not a wart. The profile could be read from the graph AST instead, as the
  shard loader already does.
- **`shard_graph.rs` hardcodes torch2nnef naming** (`model_model__{N}_inputLayernorm_…`).
  It works for the published q40ef16 Qwen/Llama exports; it is not a general mechanism.
- **`api/rs` needs an escape hatch.** This branch adds `Model::typed_ref` /
  `into_typed_model` (`#[doc(hidden)]`) because a `Model` cannot otherwise be split or
  measured. Shape it however you prefer — it is the smallest thing that unblocks this.
- **CPU needs #2477** (block-quant `AddUnicast` fusion guard) or Qwen decodes NaN. Metal
  is fine on current main (#2476, #2472, #2428 merged).
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
