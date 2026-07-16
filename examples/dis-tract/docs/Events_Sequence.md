# Event sequence: what happens, and which component does it

Traced from the code, with timings from a real 2-worker Metal run of Qwen3-8B-q40ef16
(36 layers, cut at 18). Three sequences: startup, serving a prompt, and a node dying.

Components: **coordinator** (`distract-llm`) is the zenoh **router** and the only thing that
plans; **workers** (`distract-worker`) are zenoh **clients** that each own one contiguous
layer range; the **dashboard** (`distract-dashboard`) is a client that only observes, plus a
chat box.

---

## 1. Startup: cold to serving (~49 s, of which ~40 s is shard building)

### Phase 1 — the coordinator boots and reads the model alone

`distract-llm --model X --workers 2`

1. Opens zenoh in **router** mode, listening on `tcp/127.0.0.1:7447`. Nothing else in the
   cluster resolves until this is up.
2. **Loads the entire model** and counts layers by how many `cache_key` inputs it has (36).
3. Subscribes to `distract/node/*/caps` and blocks:
   `waiting for 2 workers to advertise caps...`

The whole model is in the coordinator's RAM here. Nothing is sharded yet.

### Phase 2 — workers announce themselves

`distract-worker --name node-a --backend metal --mem-mb 4096`

4. Opens zenoh in **client** mode, connecting to the router.
5. Declares a **liveliness token** at `distract/live/{node_id}`. Zenoh retracts it the instant
   the process dies — that is how the dashboard evicts a card with no polling.
6. Loops, every 700 ms: publish `NodeCaps` (hostname, backend, `mem_budget`, cpus) on
   `distract/node/{node_id}/caps`, then query `distract/assign/{node_id}`. Nothing answers
   yet, so it keeps going.

Workers may start **before** the coordinator; the retry loop is what makes launch order
forgiving. In the reference run a worker spun for 7 s (16:27:14 -> 16:27:21) waiting for the
router to exist.

### Phase 3 — the coordinator plans (the only global decision)

7. Once `--workers` distinct node ids have advertised, it stops listening and **sorts nodes
   CPU-first, then by id**. That is why a CPU node lands on stage 0 (taking the embedding
   gather with it): stable ordering, not a placement strategy.
8. Builds a per-layer weight profile and calls `memory_weighted_cuts(profile, budgets)` ->
   cuts at `[18]`. Proportional to advertised memory only: no speed, no bandwidth, **and no
   check that a shard fits its node's budget**.
9. Declares **one queryable per node** at `distract/assign/{node_id}`, holding that node's
   `AssignSpec { stage_index, cut_layers, backend, next_hop, model_path, n_layers }` — a
   spec, **not model bytes**.
10. **`drop(full)`** — releases the model. It needed it only to count and weigh layers.

`next_hop` is where the chain is wired, by the coordinator: stage 0 gets
`distract/stage/1/in`, the tail gets `None` and therefore publishes to `distract/result`.
Workers never discover each other.

```
plan (36 layers, cuts at [18]):
  stage 0 -> node-metal-1 on metal : 2210 MiB weights (4096 MiB budget)
  stage 1 -> node-metal-2 on metal : 2183 MiB weights (4096 MiB budget)
```

### Phase 4 — each worker builds its own shard (the slow part)

11. The worker's next poll finds its spec, and it builds the shard **locally**:
    - `shard_range(cut_layers, stage)` -> e.g. layers 18..36
    - `load_shard(model_path, 18, 36, 36)` — opens the `.tgz` **on its own filesystem**,
      streams the whole 4.29 GB archive, parses and prunes the graph AST, and materialises
      **only its own ~2.2 GB** of tensors
    - `shard_io_roles(model.clone().into_optimized())` — optimises a **clone** purely to
      classify each I/O slot `Wire` or `Cache`, then discards it
    - `load_stage(..)` -> `prepare()` for its backend, which **optimises again**, applies the
      Metal transform, and uploads to the GPU
12. Subscribes to `distract/stage/{i}/in`, publishes its index to
    `distract/stage/{i}/ready`, and spawns two background tasks: a 1 s stats heartbeat on
    `distract/node/{id}/stats`, and a listener on `distract/reset`.

**This dominates startup: ~40 s** (16:27:21 -> 16:28:01). Both workers do it in parallel and
independently. The shard is optimised twice — once on the discarded clone, once inside
`prepare` — which is pure waste in the hot spot and needs no protocol change to fix.

### Phase 5 — ready

13. The coordinator collects `ready` from every stage, then declares subscribers for
    `distract/result` and `distract/resetack/*`, and the `distract/generate` queryable.

```
16:28:01  stage 0 ready (1/2)
16:28:03  stage 1 ready (2/2)
16:28:03  generation server ready on distract/generate — awaiting prompts
```

---

## 2. Serving a prompt

One at a time, strictly serial: the coordinator handles a request to completion before
looking at the next. Concurrent callers queue.

1. A client (dashboard or `distract-gen`) queries `distract/generate` with
   `GenerateRequest { prompt, max_tokens, stream_id, stop }`. Only the caller knows the
   tokenizer, so it supplies the **stop ids**.
2. Coordinator publishes to `distract/reset`; every worker clears its resident KV and acks on
   `distract/resetack/{stage}`. The coordinator waits for one ack per stage (bounded, 500 ms)
   — so a prompt can never race ahead of a reset.
3. **Per token**:
   - coordinator publishes `frame(StepMeta{turn, phase}) + tensors` to `distract/stage/0/in`
   - stage 0 runs its layers, publishes the residual to its `next_hop`
     (`distract/stage/1/in`)
   - the tail stage publishes logits to `distract/result`
   - coordinator `argmax`es, checks the token against `stop`, appends it
   - coordinator publishes `RunStats` on `distract/run` (dashboard) and the partial token list
     on `distract/stream/{stream_id}` (live chat)
4. On a stop token, or at `max_tokens`, it publishes a final `StreamMsg { done: true }` and
   replies to the query with `GenerateReply { tokens, ttft_ms, decode_tok_s }`.

Every token traverses **every** stage in sequence, so the stages compose serially: step times
add, rates do not. Two balanced stages give roughly half of one node's tok/s. Only the
residual crosses the wire; the KV never does.

---

## 3. When a node dies

Verified by killing a worker mid-generation.

**What recovers:** the dashboard evicts the card instantly on zenoh's liveliness `Delete`.
A restarted worker with the same `--name` (or the persisted `~/.dis-tract/node_id`) re-queries
`distract/assign/{node_id}`, gets **its own stage back**, rebuilds its shard (~28 s warm) and
resubscribes. KV state self-heals, because every prompt resets all caches anyway.

**What does not:** the **coordinator wedges permanently**. It declares no liveliness
subscriber, so it never learns a worker died; and its per-token wait
(`result_sub.recv_async()`) has **no timeout**. The in-flight token was published to a
subscriber that no longer exists — zenoh pub/sub is fire-and-forget, so it is simply gone —
and the coordinator blocks forever on a result that will never arrive. Because the generate
loop is serial, it never accepts another request either. Measured after the worker had fully
rejoined: a new request got no reply in 45 s, coordinator at 0.0% CPU, the rejoined worker
idle. **A healthy cluster, deadlocked.** Only restarting the coordinator clears it.

The fix is small and the pattern already exists a few lines above: bound the wait with a
`tokio::select!` deadline, as the reset-ack does, and subscribe to `znet::LIVE_WILDCARD` so a
`Delete` for a participating node aborts the current generation. Mid-generation the sequence
is unrecoverable regardless; the honest behaviour is to fail that request and accept the next.
