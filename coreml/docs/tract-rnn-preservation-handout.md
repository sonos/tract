# Handout — Preserve high-level `Gru`/`Lstm`/`Rnn` ops through tract declutter

> **What this is.** A self-contained briefing for an agent (or a human) to investigate, scope, and implement an upstream change in [sonos/tract](https://github.com/sonos/tract): keep `Gru` / `Lstm` / `Rnn` as first-class typed ops instead of decomposing them into `Scan(body=cell)` at import / declutter time.
>
> **Author context.** Written 2026-05-10 from the `tract-coreml-ane` workspace (see `notes/draft-pr-body.md` for the wider PR this would unlock).
> **Branch starting point.** `~/coding/tract-coreml-ane/tract/` (a tract checkout with our `tract-coreml` crate added). The upstream change itself targets `sonos/tract` `main`; this checkout is your reference.

---

## Why this matters — the unlock

Multiple downstream paths are blocked by the same root cause: tract decomposes RNN cells into a generic `Scan` loop, losing the high-level intent. Every accelerator backend then has to either pattern-detect (brittle, ~600–900 LoC each) or accept the unfused form and pay per-timestep dispatch overhead.

**Concrete impact across the tract ecosystem:**

| Backend | Today | With high-level RNN preserved | Estimated win |
|---|---|---|---|
| **CPU** (tract-core) | Per-op dispatch through `Scan` body, allocations between gates | Hand-fused GRU kernel: 3 input matmuls fold into one (concatenated `[W_ir\|W_iz\|W_in]`); gates stay in registers across activations | **2–5×** typical (oneDNN/MKL-class fusion) |
| **tract-metal** | Per-timestep Metal kernel launch for each cell op | Translate `Gru` → `MPSCNNRNN` (Metal Performance Shaders has dedicated RNN ops); single GPU dispatch covers the whole sequence | **3–10×** depending on T |
| **tract-cuda** | Per-timestep CUDA launch | Translate `Gru` → `cudnnRNNForward` (NVIDIA's flagship RNN kernel) | **2–10×** |
| **tract-coreml** (proposed PR) | Scan body stays on CPU | Translate `Gru` → `mb.gru` → ANE-eligible | DFN3 RTF 0.066 → likely **0.02–0.03**; potential ANE engagement |

**Three canaries currently affected:**

| Canary | Format | Sub-model | Stranded today |
|---|---|---|---|
| **DFN3** (DeepFilterNet 3) | NNEF | decoder | 2× GRU `tract_core_scan` |
| **Parakeet TDT v3** (NVIDIA ASR) | NNEF | decoder | LSTM scan (15 CoremlOps; recurrent body on CPU) |
| **Nemotron Speech Streaming** | NNEF | decoder | LSTM scan (matches Parakeet structurally) |

All three sit in Sonos's own CI sweep — so this isn't an ask that benefits external users at the maintainer's expense; **the maintainer is the largest beneficiary**.

**Secondary wins** (less obvious but real):
1. **Quantization becomes tractable.** RNN quantization needs awareness of hidden-state scale across timesteps — a decomposed scan body doesn't carry that context. Calibration tools get the structure they need.
2. **Compilation/declutter passes get faster.** Less IR to traverse; multi-layer LSTM stacks see noticeable load-time wins.
3. **ONNX/NNEF round-trip preserves structure.** Today: import as Gru → declutter to Scan → can't export back as Gru. With preservation, format conversions stay clean.
4. **Profiling becomes readable.** A single `Gru` op that took 8 ms is much easier to read than `Scan(GRUCell)` × 100 timesteps × 12 ops/cell = 1200 line items.

---

## What changes (the proposal in one paragraph)

Add `Gru` / `Lstm` / `Rnn` as high-level typed ops in `tract-core` (parallel to how `Conv` already works). Change the ONNX importer to emit them directly. Add a new public model transform `lower_rnn_to_scan` that performs the current decomposition. Make `into_decluttered()` call it by default — so the **CPU runtime sees no change in behavior**. Backend transforms (tract-coreml, tract-metal, tract-cuda) opt out by running their own transform *before* `into_decluttered()`, then translate the surviving high-level RNN op to their accelerator's RNN intrinsic.

This is a **transform-ordering refactor, not a semantics change.** Default behavior unchanged for any code path that calls `into_decluttered()` then runs CPU.

---

## Where tract sits today (concrete file map)

### ONNX import (currently goes straight to Scan)

| File | LoC | Role |
|---|---:|---|
| `onnx/src/ops/rec/common.rs` | 445 | `WireBody` trait + `CommonRec` struct that wires the body into a `tract_core::ops::scan::Scan` |
| `onnx/src/ops/rec/gru.rs` | 150 | `impl WireBody for GRU` — wires the GRU cell body (3 matmuls + sigmoids + tanh + gate combine) into a Scan body |
| `onnx/src/ops/rec/lstm.rs` | 183 | Same pattern for LSTM (4 gates + cell state) |
| `onnx/src/ops/rec/rnn.rs` | (small) | Vanilla RNN |
| `onnx/src/ops/rec/scan.rs` | — | Generic ONNX `Scan` op (separate from the RNN-cell wiring) |

Key trait (`onnx/src/ops/rec/common.rs:8`):
```rust
pub trait WireBody: Debug + DynClone + Send + Sync {
    fn name(&self) -> &'static str;
    fn wire_body(&self, prefix: &str, body: &mut TypedModel) -> TractResult<()>;
    fn w_b_multipliers(&self) -> (usize, usize);
    fn have_extra_c_state(&self) -> bool;
}
```

`CommonRec::wire_as_scan` (around line 232 of common.rs) calls `self.body.wire_body(prefix, &mut body)` and constructs the `Scan` directly. This is the spot where the high-level intent gets lost.

### tract-core (where the new Gru/Lstm ops would live)

| File | LoC | Role |
|---|---:|---|
| `core/src/ops/scan/mod.rs` | 95 | `ScanInfo` struct |
| `core/src/ops/scan/decluttered.rs` | (large) | `pub struct Scan` — the typed-model RNN/scan op |
| `core/src/ops/scan/optimized.rs` | (large) | Optimized form for codegen |

There is **no `Gru` / `Lstm` typed op in tract-core today.** This is the asymmetry being fixed. (Compare with `core/src/ops/cnn/conv/` which preserves Conv as a typed op all the way through.)

### NNEF (different story — already loses structure at the format level)

NNEF doesn't have GRU/LSTM as primitives — Parakeet/Nemotron NNEF dumps encode their RNNs as `tract_core_scan` directly. So preserving the op only helps **on-import**, not for already-NNEF-encoded models. To benefit Parakeet/Nemotron, an additional pass would be needed:

> **NNEF-side option** (out of scope for the first PR): a pattern-detector that recognizes `scan(GRUCell|LSTMCell pattern)` at import time and re-folds it into `Gru`/`Lstm`. Defer; ship ONNX-side first.

### Existing decluttered code that depends on Scan structure

| File | Why it matters |
|---|---|
| `core/src/ops/scan/decluttered.rs` | Self — declutter rules on Scan itself |
| `core/src/optim/slice.rs` | Slice-through-Scan optimization |
| (potential) any quantization/profiling passes | Need to grep — these may pattern-match on Scan body shape |

**Investigation task:** grep tract-core for `op_as::<Scan>()` and `downcast_ref::<Scan>` to find all consumers. Each one needs to be checked: does it work with `Gru` directly, or does it require seeing the lowered Scan body?

---

## Compatibility strategy (zero behaviour change for CPU users)

The cleanest shape:

```rust
// New public transform.
let lower_rnn = get_transform("lower_rnn_to_scan")?;

// Default declutter calls it as the last step (no behaviour change for current users):
fn into_decluttered(self) -> TractResult<TypedModel> {
    // ... existing declutter passes ...
    let lower_rnn = get_transform("lower_rnn_to_scan").unwrap().unwrap();
    let mut m = self;
    lower_rnn.transform(&mut m)?;
    Ok(m)
}

// Backend authors opt out:
let mut model = onnx_model.into_typed()?;  // sees high-level Gru
CoremlTransform::default().transform(&mut model)?;  // emits mb.gru for ANE
// Backend may then run lower_rnn for any residual Gru that didn't translate:
get_transform("lower_rnn_to_scan").unwrap().unwrap().transform(&mut model)?;
```

Rules:
- **Public API unchanged.** No new feature flags. Just a new transform name.
- **CPU users** keep getting the same Scan-decomposed model. `into_decluttered()` runs `lower_rnn_to_scan` internally.
- **Backend authors** call `transform` *before* `into_decluttered`, or override the declutter path.
- **Forward-compat for backends that don't recognize `Gru`:** the model still works because they'd run `lower_rnn_to_scan` afterward and fall through to Scan. (Same pattern as Conv: backends that don't recognize Conv lower it to Im2Col + MatMul.)

---

## What to investigate before writing code

In this order:

1. **Where do Scan consumers live?** Grep tract for `op_as::<Scan>` / `downcast_ref::<Scan>`. Each consumer needs to be read and classified: (a) safe to ignore high-level Gru (operates on Scan only post-lowering) or (b) must also handle Gru.
2. **Are there CPU optimizer passes that fuse across Scan body ops?** If so, the "no CPU behaviour change" guarantee depends on those passes still seeing the lowered form. Confirm they run *after* `lower_rnn_to_scan`.
3. **Does any quantization code introspect the cell body?** Sonos uses tract for quantized inference at the edge. If the quantizer needs to walk the body to set per-gate scales, the high-level Gru op needs to expose the same information (e.g., as per-gate weight slices).
4. **What does TensorFlow's `block_lstm` do** (`tensorflow/src/ops/rec/block_lstm.rs`)? Likely a separate codepath that already preserves a high-level LSTM op for TF — could be the reference model.
5. **What does the maintainer think?** This is the most important step. Open a GitHub Discussion or issue on `sonos/tract` *before* writing any code. The proposal touches enough of the IR that getting buy-in early avoids wasted work.

---

## What to implement (when buy-in is secured)

Roughly in dependency order, with effort estimates:

### Phase 1 — Add the typed ops (small, no behaviour change)

- Add `core/src/ops/rnn/{mod,gru,lstm,rnn}.rs` — typed op definitions with `Op + EvalOp + TypedOp` impls
- `EvalOp` impls can delegate to a Scan-lowering at construction time (so CPU eval works while a real fused kernel doesn't yet exist)
- Add `core/src/ops/rnn/lower_to_scan.rs` — the transform that rewrites `Gru` / `Lstm` / `Rnn` to `Scan(body=cell)` (bulk of code is moved from `onnx/src/ops/rec/common.rs::wire_as_scan` and the per-cell `wire_body` impls)
- Register the transform: `register_transform!("lower_rnn_to_scan", ...)`

**Estimate:** ~600–800 LoC mostly moved, not new logic.

### Phase 2 — Switch ONNX importer to emit high-level ops (small)

- Change `onnx/src/ops/rec/{gru,lstm,rnn}.rs` so they emit the new `tract_core::ops::rnn::Gru` (etc.) directly instead of constructing a `Scan`
- The existing per-cell `wire_body` impls become **only** the lowering code in Phase 1 — they're no longer the import path

**Estimate:** ~150 LoC delta.

### Phase 3 — Make `into_decluttered()` call `lower_rnn_to_scan` (tiny)

- Append `get_transform("lower_rnn_to_scan").unwrap().unwrap().transform(&mut model)?` as the last declutter step
- Add a doc comment explaining the contract: "if you want to see high-level RNN ops, run your transforms before `into_decluttered()`"

**Estimate:** ~30 LoC.

### Phase 4 — Tests (essential)

- `core/src/ops/rnn/test.rs` — round-trip test: build a Gru, lower to Scan, run, compare to the same Gru's CPU eval
- `onnx/src/ops/rec/test.rs` — load ONNX GRU/LSTM, verify high-level op survives if `lower_rnn_to_scan` is skipped
- Numerical correctness: check that lowering produces a Scan whose CPU eval matches ONNX Runtime's GRU/LSTM output for a fixed seed

**Estimate:** ~200–400 LoC.

### Phase 5 — Optional fused CPU kernel (large, defer)

- Hand-tuned `Gru::eval` that fuses gate matmuls, avoids per-step allocations
- Out of scope for the first PR — Phase 1's lowering-based eval is correctness-equivalent. Optimization is a follow-up.

### Phase 6 — Optional NNEF importer pattern detector (large, defer)

- Walk `tract_core_scan` bodies, recognize GRUCell / LSTMCell patterns, re-fold into `Gru` / `Lstm`
- Out of scope for first PR. Defer until needed (likely when targeting Parakeet/Nemotron specifically).

**Total Phase 1–4 estimate:** ~1000–1400 LoC, mostly moved. ~2–4 days of focused work for someone fluent in tract's IR.

---

## Risks and open questions to raise with the maintainer

- **Does the existing TF `block_lstm` path conflict?** If TF already has a high-level LSTM op, do we converge on its design, or do they coexist?
- **Multi-layer / bidirectional / variable-sequence-length** — what's the canonical representation? ONNX has these as op attributes; MIL has them as separate ops. The high-level `Gru` op should support all three.
- **Initial hidden state** — input or attribute? ONNX makes it an optional input; we should match.
- **Quantization** — Sonos uses tract for quantized RNNs at the edge. Does the high-level op need to surface per-gate quantization params? (Almost certainly yes.)
- **What's the deprecation story** for code that pattern-matches on the post-lowered Scan? Probably none — `lower_rnn_to_scan` makes Scan still appear by default, so old code keeps working.
- **Bidirectionality storage** — how do we represent the two directions? ONNX uses `direction` attribute + duplicated weights. MIL has separate `mb.gru` ops for forward/backward. The high-level `Gru` op should probably mirror ONNX's convention (single op + direction attribute) and let backends split if needed.

---

## Evidence to bring to the upstream conversation

1. **Three Sonos-CI canaries blocked by this** (DFN3, Parakeet decoder, Nemotron decoder) — see `notes/phase-4-dfn3-closure.md` for the DFN3 detail and the perf table at `notes/draft-pr-body.md` for the multi-canary view.
2. **Per-mode latency analysis on RVM** (`notes/draft-pr-body.md`, "ANE engagement" section) showing that recurrent loops blow ANE engagement even after we fix the obvious bugs — same root cause class.
3. **Multi-backend leverage:** the same fix benefits CPU + Metal + CUDA + CoreML simultaneously. One upstream change → four downstream wins. For Sonos this is 4× leverage on their own engineering.
4. **Existing precedent:** `Conv` is preserved as a typed op all the way through declutter; backends that recognize Conv emit it directly, others lower it. RNN preservation is just applying the same architectural pattern.
5. **Open question Q1 in [`notes/tract-upstream-feedback.md`](./tract-upstream-feedback.md)** ("Why does tract decompose high-level ops aggressively at declutter time?") is the broader version of this. RNN preservation would be the first concrete instance of a possibly-larger policy shift.

---

## Recommended first step

**Open a GitHub Discussion on `sonos/tract`** titled something like *"Proposal: preserve `Gru`/`Lstm`/`Rnn` as high-level typed ops through declutter"*.

Body should include:
- The summary paragraph above
- The multi-backend benefit table
- The transform-ordering compat strategy (with the code sketch)
- Pointers to all three affected canaries with concrete CoremlOp/perf numbers
- Explicit list of compatibility risks and questions for the maintainer

Wait for buy-in (or pushback that reveals constraints we don't see) **before** writing any code. The proposal touches enough of the IR that aligning with the maintainer's view of the architecture matters more than landing fast.

---

## Out of scope for this handout

- Implementing the per-backend MIL/Metal/cuDNN emit (that's downstream work in each backend crate)
- Quantization design for the new ops (separate proposal)
- Optimized fused CPU kernels (Phase 5, deferred)
- NNEF importer pattern detection (Phase 6, deferred)

---

## Files an agent should read in this order

1. **`notes/tract-upstream-feedback.md`** — context on what we've already documented for the maintainer; especially I4 (decompose-aggression) and Q1 (open architectural question)
2. **`notes/draft-pr-body.md`** — the wider PR this would unlock; perf tables; ANE engagement matrix
3. **`notes/phase-4-dfn3-closure.md`** — primary DFN3 evidence; the existing analysis of why GRU stays on CPU and the three approaches considered
4. **`tract/onnx/src/ops/rec/common.rs`** — the wiring code that needs to be split (import-emit vs. lower-to-scan)
5. **`tract/onnx/src/ops/rec/gru.rs`** + `lstm.rs` — the per-cell `wire_body` implementations
6. **`tract/core/src/ops/scan/decluttered.rs`** — what Scan looks like today (the lowering target)
7. **`tract/core/src/ops/cnn/conv/`** — the architectural precedent for "preserve high-level op through declutter"
