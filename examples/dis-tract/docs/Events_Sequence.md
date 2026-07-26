# Event sequence: what happens, and which component does it

Traced from the code, with timings from a real 2-worker Metal run of Qwen3-8B-q40ef16
(36 layers, cut at 18). Three sequences: startup, serving prompts, and a node dying.
Throughput figures in section 2 come from a separate CPU+Metal run of Qwen2.5-7B-q40ef16 and
are labelled where they appear.

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
2. **Parses the NNEF graph AST** from the archive — reads the `graph.nnef` text and the
   `.dat` file sizes in one pass, counts layers by how many `cache_key` parameters it has
   (36), and computes the per-layer weight profile from the AST (which `variable` labels
   feed which layer's cache). No `TypedModel` is ever loaded.
3. Subscribes to `distract/node/*/caps` and blocks:
   `waiting for 2 workers to advertise caps...`

The model's weights are never in the coordinator's RAM — only the graph text and the
`.dat` file sizes are read from the archive.

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
   cuts at `[18]`. Proportional to advertised memory only: no speed, no bandwidth. The split
   is checked against the budgets, so a stage that cannot fit its node fails **here**, before
   any worker spends ~40 s building a shard it cannot hold.
9. Declares **one queryable per node** at `distract/assign/{node_id}`, holding that node's
   `AssignSpec { stage_index, cut_layers, backend, next_hop, model_path, n_layers }` — a
   spec, **not model bytes**.

There is nothing to release afterwards: the coordinator never held the model, only the graph
text and the `.dat` sizes.

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
    `distract/stage/{i}/ready`, and spawns three background tasks: a 1 s stats heartbeat on
    `distract/node/{id}/stats`, a listener on `distract/reset/*` that clears **one
    sequence's** KV and acks, and a listener on `distract/free/*` that drops a finished
    sequence's KV outright.

**This dominates startup: ~40 s** (16:27:21 -> 16:28:01). Both workers do it in parallel and
independently. The shard is optimised twice — once on the discarded clone, once inside
`prepare` — which is pure waste in the hot spot and needs no protocol change to fix.

### Phase 5 — ready

13. The coordinator collects `ready` from every stage, then declares subscribers for
    `distract/result` and `distract/resetack/*`, and the `distract/generate` queryable.

```
16:28:01  stage 0 ready (1/2)
16:28:03  stage 1 ready (2/2)
16:28:03  generation server ready on distract/generate — up to 2 sequences in flight
```

---

## 2. Serving prompts

Several at a time. Each request becomes a **sequence** with its own KV slot on every stage,
and their steps interleave down the one pipeline. Admission is bounded by `--max-sequences`
(default: one per stage); past that, queries wait in the queryable.

1. A client (dashboard or `distract-gen`) queries `distract/generate` with
   `GenerateRequest { prompt, max_tokens, stream_id, stop }`. Only the caller knows the
   tokenizer, so it supplies the **stop ids**.
2. The coordinator assigns a sequence id and publishes to `distract/reset/{seq_id}`; every
   worker clears **that sequence's** KV and acks on `distract/resetack/{stage}` with
   `[stage, seq_id]`. The coordinator waits for one ack per stage (bounded, 500 ms) — so a
   prefill can never race ahead of its own reset. Sequences already in flight are untouched.
3. **Per step**, for each sequence not already waiting on a result:
   - coordinator publishes `frame(StepMeta{turn, phase, seq_ids}) + tensors` to
     `distract/stage/0/in`
   - stage 0 runs its layers against the KV for `seq_ids[0]`, publishes the residual to its
     `next_hop` (`distract/stage/1/in`), echoing the `StepMeta` unchanged
   - the tail stage publishes logits to `distract/result`
   - coordinator reads the sequence id back out of the echoed `StepMeta`, `argmax`es, checks
     the token against that sequence's `stop`, appends it
   - coordinator publishes `RunStats` on `distract/run` (dashboard) and the partial token list
     on `distract/stream/{stream_id}` (live chat)
4. On a stop token, or at `max_tokens` generated tokens, it publishes a final
   `StreamMsg { done: true }`, replies to that sequence's query with
   `GenerateReply { tokens, ttft_ms, decode_tok_s, error }`, and publishes
   `distract/free/{seq_id}` so the stages drop its KV.

Results are **matched** by sequence id, not assumed to arrive in the order the steps went
out. A result naming a sequence that is no longer active is dropped, never applied to
whichever sequence next reuses the slot.

With `--prefill-chunk N`, a prompt is fed N tokens per step instead of all at once. Only the
last chunk's logits are a token; the rest predict something the prompt already contains. It
bounds how long one long prompt can hold a stage, and costs one extra pipeline traversal per
chunk — worth it when serving mixed prompt lengths, a straight loss when serving one client.

### What interleaving can and cannot buy

A sequence's steps are strictly ordered — token `t+1` is the argmax of step `t` — so one
sequence never occupies more than one stage, and **per-sequence latency does not improve**.
What improves is cluster throughput: while sequence A's step is in stage 1, stage 0 takes B
instead of idling. Only the residual crosses the wire; the KV never does.

The ceiling is set by the stages, not the scheduler. With step times `t0..tn`, interleaving
can reach `(Σti)/max(ti)` and no more — the slowest stage is always busy. Measured on
Qwen2.5-7B-q40ef16 across a CPU stage 0 and a Metal stage 1, two concurrent sequences against
the same two run back-to-back:

| cut | stage 0 (cpu) | stage 1 (metal) | ceiling | measured |
|-----|---------------|-----------------|---------|----------|
| 14  | 514.2 ms      | 44.6 ms         | 1.09x   | 1.12x    |
| 5   | 261.7 ms      | 65.2 ms         | 1.25x   | 1.26x    |

Both are at the ceiling, so the headroom is **the split, not the scheduler**.
`memory_weighted_cuts` balances bytes — right for making a model fit, wrong for throughput
when the stages run on backends of different speed. Moving the cut from 14 to 5 by advertised
budgets alone, with no code change, nearly halved serial wall time. A planner that balanced
time would need nodes to advertise throughput as well as memory.

---

## 3. When a node dies

Verified by killing a worker mid-generation.

**What recovers:** the dashboard evicts the card instantly on zenoh's liveliness `Delete`.
A restarted worker with the same `--name` (or the persisted `~/.dis-tract/node_id`) re-queries
`distract/assign/{node_id}`, gets **its own stage back**, rebuilds its shard (~28 s warm) and
resubscribes. KV state self-heals: a new sequence resets its own slot before its first step,
and a slot left behind by a sequence nobody finished is reclaimed once it goes idle.

**What the coordinator does:** it subscribes to `distract/live/*`, so zenoh retracting the
dead worker's token ends the in-flight generations at once — no timeout to wait out. They are
unrecoverable (the step in flight was published to a subscriber that no longer exists, and
zenoh pub/sub is fire-and-forget). The stages are **shared**, so a stage that dies takes down
every sequence riding the pipeline, not only the one whose step was outstanding: the
coordinator abandons all of them, replies to each with its partial tokens and an `error`,
frees each slot, and goes back to serving. The wait is also bounded (`STEP_TIMEOUT`, 30 s) to
cover a stage that is merely wedged rather than gone.

The node-death path was verified by killing a worker mid-generation, before sequences could
interleave; the message then named the turn rather than the sequence. What has been observed
since is the other end of the same code path — the deadline, under memory pressure, with one
sequence in flight:

```
ERROR pipeline stalled, abandoning 1 sequence(s): no stage reply in 30s
WARN  seq 2 abandoned after 34 tokens
```

Both paths call the same abandon-everything routine; the multi-sequence node-death case has
not been re-run since.

then restarting it: `assigned stage 1 on metal`, and the next prompt answered normally
(`generated 11 tokens, 20.6 tok/s`) with no coordinator restart. The stages' KV is out of
step after an abort, but the reset at the top of the next prompt clears every stage, so the
cluster stays usable.

