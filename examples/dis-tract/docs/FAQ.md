# Dis-tract FAQ

Answers to what people actually ask, worst news first.

## Status

### Does Dis-tract run a model too big for one machine?

Not yet. That is the goal, and it is **not demonstrated**. Every run so far is two worker
processes on a *single* machine over loopback, on Qwen3-8B-q40ef16 (4.29 GB) — a model that
fits that machine on its own.

What *is* exercised is the mechanism: each worker prunes the NNEF graph to its own layer
range and reads only that shard's tensors, so the full model is never materialised in a
worker; the coordinator's plan splits 36 layers into 2210 MiB + 2183 MiB shards; and the
generated tokens match a single-machine reference exactly.

The experiment that would earn the claim is two 16 GB machines and a model needing both
(Qwen3-32B, ~17.6 GB). One thing blocks it beyond hardware: the coordinator still loads the
whole model to compute its layer weight profile before dropping it, so it needs a box that
fits the model — precisely what the too-big case does not have. That profile should come
from the graph AST, as the shard loader already does.

### Why greedy decode, and no memory of the last turn?

Deliberate scope. Sampling is argmax, so a run is reproducible and can be checked
token-for-token against a single-machine reference — that parity is the property the whole
thing is validated on, and a sampler would destroy it. The KV is cleared before each prompt,
so there is no multi-turn context by design.

Neither is load-bearing for the split; both are what a PoC leaves out. See also *Can it
serve more than one request at a time?* — the single-stream limit is the one with real
consequences.

### Which models work?

The q40ef16 NNEF exports tract CI publishes (Qwen3-8B, Qwen3-1.7B, Llama-3.2-1B, …).

`shard_graph.rs` finds layer boundaries by matching torch2nnef's naming
(`model_model__{N}_inputLayernorm_…`), so it is Qwen/Llama-shaped exports only, not a
general mechanism. A model whose export names things differently will not shard.

## Running it

### Why must the coordinator start first, and why the hardcoded `tcp/127.0.0.1:7447`?

The coordinator is the zenoh **router**; workers and the dashboard are **clients** that
route through it. Clients retry, so order is forgiving in practice, but nothing resolves
until the router listens.

On a real LAN, zenoh **scouting** discovers the router with no addresses configured. On
macOS, loopback multicast is unreliable, so localhost bootstraps from a fixed endpoint
instead. A consequence worth stating plainly: because everything so far has run on one
machine, **the scouting path is untested**.

### The dashboard says the cluster does 15 tok/s, but each node says 25. Which is right?

Both — they measure different things.

This is **pipeline** parallelism, not data parallelism. The nodes do not work on different
tokens in parallel; every token passes through *every* stage in sequence. So the stages
compose serially:

```
token -> stage 0 (36 ms) -> stage 1 (34 ms) -> done      = 70 ms/token = 14.3 tok/s
```

The combining rule is reciprocal, not additive: `1/(1/27.7 + 1/29.4) = 14.3`. **Two balanced
stages give roughly half of one node's number.** A node reporting "27.7 tok/s" means *"I
could sustain that for my own half"* — a rate no token ever achieves, because it also needs
the other half. Step times add; rates do not.

The cluster banner also averages over the whole run while the node cards show the latest
step, and steps slow as the KV cache grows — so the average sits above the current rate and
drifts down. The banner shows both (`avg` and `now`).

### Can it serve more than one request at a time?

No. **Single stream, strictly serial.** The coordinator is one loop over its
`distract/generate` queryable: it takes a request, clears every worker's KV, decodes that
prompt to completion, replies, and only then looks at the next. Concurrent requests do not
error — they queue, and a caller just waits. There is no batching, and each prompt starts
from an empty cache, so there is no multi-turn context.

This is worth being pointed about, because a pipeline is exactly where it costs you. With a
single stream, only one stage is busy at a time: while stage 0 computes token *n*, stage 1
sits idle, and vice versa. In a 2-stage split each node is therefore ~50% idle, and that
does not improve with more nodes — an N-stage pipeline leaves each node ~1/N busy. The
standard answer is **micro-batching** (keep several sequences in flight so every stage
always has work), which is what makes pipeline parallelism pay for throughput. It is not
implemented. Single-stream *latency* is unaffected by this — the split costs ~1% — but the
hardware is mostly idle, and that is the obvious next win for a serving workload.

### Could this run over the internet, not just a LAN?

Mechanically, zenoh would not stop you: it speaks TCP and QUIC and routers can be remote. In
practice, no — as it stands it is a LAN design, for two reasons.

**Latency, not bandwidth, is the problem.** The payload is tiny: the residual is one
`[1, 1, hidden]` f16 tensor, ~8 KB per token for a 4096-wide model. Bandwidth is a
non-issue. But it is *per token, per stage boundary, serially* — the next token cannot start
until the previous one finishes the whole chain. At ~43 ms/step, a 30 ms round-trip roughly
doubles per-token latency; a 100 ms link triples or worse. Loopback hides this entirely,
which is why the measured "splitting costs ~1%" carries no wire cost at all.

**Nothing here is safe to expose.** The endpoint is plain `tcp/`, with no TLS, no
authentication, and no authorisation on the key space: any peer that can reach the router
can publish activations onto a stage's input key or answer an assignment query. zenoh
supports TLS/QUIC and access control; none of it is configured.

If you wanted this over a WAN, the honest shape is different — batch aggressively to amortise
the round-trip, or cut in fewer places, and turn on zenoh's transport security first.

### What do the dashboard's memory figures mean?

- **shard weights** — bytes of model weights this shard holds, counted *as stored*:
  block-quant (q40) weights are counted packed, not at their dequantized f32 size. These sum
  to the full model across the cluster.
- **process memory** — the physical memory the worker process occupies. On macOS this is
  `phys_footprint`, not resident-set size: RSS omits compressed and GPU-driver pages and
  reads ~11 MB for a worker whose real footprint is ~2.4 GB.
- **host memory** — the whole machine, every process. Co-located workers report the same
  figure, so it must not be summed across them.

### Do I need any unmerged PRs?

For CPU workers, yes: without sonos/tract#2477 (block-quant `AddUnicast` fusion guard) Qwen
decodes NaN. Metal is fine on current main. Check whether #2477 has merged before believing
this line.

## Performance

### Does splitting cost throughput?

Barely, in-process: a 2-shard chain matches the whole model. Qwen3-8B-q40ef16, Metal,
`distract-shardbench` (min of 14, so the only variable is how the model was built):

| config | ms/step | tok/s |
|---|---|---|
| whole model (`load_model`) | 42.5 | 23.5 |
| full-range shard, **unsplit** | 42.4 | 23.6 |
| **2-shard chain** | 43.0 | 23.3 |

Token-identical across all three; ~1%.

The caveat is the whole point: that is measured **with no wire between the stages**. A live
2-worker cluster on one machine over loopback does 21–22 tok/s. On a real network each token
buys a round-trip per stage boundary — at ~43 ms/step even a 1 ms hop is ~2%, and a slow
link is worse. Only the residual crosses (never the KV), so the payload is small; small is
not zero.

### My shard is much slower on Metal but identical on CPU. What did I break?

Almost certainly a missing **declutter**.

`tract_nnef::nnef().translate(&proto, ..)` returns a **raw** model. The high-level `api/rs`
loader is `model_for_path(p)?.into_decluttered()?` — it decluttes; `translate` does not. Any
code that builds a model from a `ProtoModel` (a pruned or rewritten graph, say) silently
skips it.

The result is invisible on CPU and brutal on GPU, because the GPU transforms' rewrite rules
only match decluttered patterns. Measured on Qwen3-8B: 5364 nodes undecluttered vs 1690
decluttered — same semantics, identical tokens — costing 117.8 ms/step vs 42.4 on Metal
(2.8×), with 73 `MultiBroadcastTo` stranded on the host instead of becoming
`GpuMultiBroadcastTo`, plus 146 stray host `Slice`s. CPU was 271 vs 306 ms, i.e. unaffected.

`distract-metalaudit <model> <n_layers>` prints device-vs-host op placement and node counts,
which is how to confirm it.

### Why is decode slower than a single node running the whole model?

For a single stream it should not be, much — see the table above. If a live cluster is
slower than the in-process chain, suspect the things the chain does not have: the per-token
wire hop, both stages contending for one GPU when co-located, and (at depth) the KV cache
growing.

## Design

### Why doesn't the KV cache cross the wire?

Each worker keeps its own layers' KV resident and loops it step→step; only the residual
activation crosses a stage boundary. The cache grows with context and would dominate the
wire; the residual is a single `[1, S, hidden]` tensor.

The mechanism is a role per I/O slot (`protocol::Role`): `Wire` slots (residual, token ids,
logits) cross; `Cache` slots are seeded empty, fed back from the worker's own state each
step, and never sent. This is why cuts are taken on the residual edge between layers — a cut
that separated a layer from its cache would put the cache on the wire. (The core primitive's
own guard is more general: `extract_subgraph` refuses a cut whose output cone reaches a
Source you did not declare as a boundary input.)

### How are shards assigned to nodes — by memory, by speed, or both?

**Memory only.** Nothing about performance is modelled, and there is no fit check.

The flow: each worker advertises a `mem_budget` on join (`--mem-mb N`, or `--mem-frac` of
its available RAM). The coordinator builds a per-layer weight profile of the model, then
`plan::memory_weighted_cuts` walks the layers placing cuts so that **cumulative weight is
proportional to cumulative budget** — a node advertising twice the budget gets roughly twice
the weight. Weights are counted as stored, so q40 blocks count packed, not at f32 size.

What that means in practice:

- **A fast node and a slow node with equal budgets get equal layers.** Backend, core count,
  and memory bandwidth are not inputs, so a Metal node and a CPU node with `--mem-mb 4096`
  each get half the model — and since every token traverses both, the pipeline runs at the
  slower node's pace. If you want to bias work toward a faster node today, you do it by
  hand, by lying about budgets.
- **Link bandwidth and latency are not inputs either**, so the planner cannot prefer a cut
  that minimises what crosses a slow hop.
- **There is no fit check.** The split is proportional, full stop: if a node's share exceeds
  the budget it advertised, it is still assigned it and fails (or swaps) at load time
  instead of at plan time. Proportional-and-silent was survivable while the weight figures
  were themselves wrong by ~8x; now that they are accurate, a real check is cheap and
  worth adding.
- **Stage order is not a decision.** Nodes are sorted CPU-first, then by node id, so a CPU
  node takes stage 0 (and with it the embedding gather). That is a stable-ordering
  convenience, not a placement strategy.

A profile-guided planner — measure each node, weight by throughput and link cost, verify the
fit — is the obvious next step and is not implemented.

### Why isn't the split a registered `ModelTransform`?

This is the open design question, and the conflict looks real rather than a matter of
effort. A `ModelTransform` operates on an **already-loaded `TypedModel`** — so the full
model must be materialised before it can be split, which is exactly what a
too-big-for-one-machine split has to avoid.

Dis-tract instead prunes the **NNEF graph AST** and reads only the shard's `.dat` tensors
(EXO's approach), which needs loader internals `api/rs` does not expose. So the upstreamable
primitive is probably not the splitter but a **public partial-load API** for NNEF — load a
graph subset plus only its tensors. With that, Dis-tract becomes an ordinary
`causal_llm`-style example on the public API.

### Why zenoh?

Because a cluster needs three different things and zenoh is one dependency that does all
three, rather than three hand-rolled ones:

- **discovery** — scouting: a worker finds the coordinator with no addresses configured
- **the hot path** — pub/sub, for the per-token residual
- **assignment** — a queryable the coordinator serves, which each worker pulls by its own id

It is also EXO's own transport, which matters here: EXO solved this exact problem
(heterogeneous boxes on a home LAN), so borrowing its transport means borrowing decisions
already validated at that scale. It is on crates.io under Apache-2.0, so it is a normal
dependency rather than a vendored fork.

The codec is transport-agnostic (frames over any `Read`/`Write`), so this is a swappable
choice, not a foundation. An earlier revision ran the same protocol over HTTP; moving to
zenoh touched only the two binaries. Raw framed TCP would be lighter if discovery were not
needed.

Three things worth knowing if you touch it:

- zenoh needs a **multi-threaded** tokio runtime — `#[tokio::main]` on `current_thread`
  panics.
- **Peer mode floods** worker-to-worker connection attempts across every interface address
  on a multi-homed host, up to fd exhaustion. Hence the star: the coordinator is the
  **router**, workers are **clients** that route through it.
- macOS **loopback multicast is unreliable**, so localhost cannot scout and bootstraps from
  a fixed endpoint instead. Everything so far has run on one machine, so the scouting path —
  zenoh's main draw — is the least tested part of this.

### Is tensor parallelism supported?

No. Pipeline/layer parallelism only. Tensor parallelism (sharding *within* a layer, plus a
collective) is a documented hook: it would reuse the same partition machinery and the same
worker, and the step channel already carries a generic `TVec<Tensor>` rather than named
slots, so a collective phase needs no new message type. It is not implemented.

### Why are there 13 binaries?

Four are the product — `distract-llm` (coordinator), `distract-worker`, `distract-dashboard`,
`distract-gen` (headless client). The other nine are diagnostics that earned their keep
finding the bugs above, chiefly `distract-shardbench` (whole vs shard vs split, in one
process) and `distract-metalaudit` (device-vs-host placement). They are the reason the
declutter regression was traced to the shard *builder* rather than to the split.
