# tract-coreml

A tract backend that dispatches supported subgraphs of a `TypedModel` to
[Apple Core ML](https://developer.apple.com/documentation/coreml) on macOS / iOS,
with the goal of engaging the Apple Neural Engine (ANE) where possible.

Sibling to [`tract-metal`](../metal) and [`tract-cuda`](../cuda) — registered
through the standard `tract_core::Runtime` trait, opt-in via workspace member
`coreml`. Not in `default-members` (macOS-only).

## Status (2026-05-10, Phase 4 + transformer-prep)

### Canary perf table

| Model | Class | Input | Inference | RTF | ANE | CoremlOps |
|---|---|---|---:|---:|---|---:|
| MobileNet v2 | CNN classifier | 224×224 | 3.4 ms | 0.103 @30fps | ✅ 185 mW | 2 |
| SqueezeNet 1.x | CNN classifier | 224×224 | 0.7 ms | 0.022 @30fps | (too small) | — |
| MODNet | CNN segmentation | 512×512 | 16.1 ms | 0.97 @60fps ✓ | ✅ 200–600 mW | 2 |
| RVM | recurrent video matting | 480×640 | 9.9 ms | 0.59 @60fps ✓ | ❌ CPU+GPU only | 2 |
| **SAM 2 Hiera-Tiny encoder** | **transformer (windowed attn)** | **1024×1024** | **1.62–2.00 s** | (interactive) | ❌ ANE compile fails | **39–47** |
| **DFN3 decoder** | **NNEF, GRU + scan** | **1 s audio chunk** | **66.4 ms** | **0.066** ✓✓✓ | (small subgraphs) | **3** |

### Op coverage: ~25 translators

- **Convolution / pooling**: Conv, MaxPool, AvgPool (via SumPool detection), Pad
- **BinOp**: Add, Mul, Sub, Min, Max
- **Activation**: Sigmoid, Tanh, Silu, HardSwish, Gelu, GeluApproximate, Erf, Rsqrt, Sqrt, Square
- **Norm**: InstanceNorm fold (multi-node detector), LayerNorm fold (multi-node detector), RmsNorm standalone
- **Transformer**: Softmax, general MatMul (runtime×runtime + multi-K/M/N + strip + expand + batch reorder + rank-6 boundary strip)
- **Shape**: Reshape, MoveAxis (Transpose), AddAxis, RmAxis, Slice, Concat, MultiBroadcastTo, Resize
- **Numerical**: Cast, Reduce (Sum/Max/Min/Prod/MeanOfSquares), EinSum (5 conv-shaped patterns + general matmul)

### Build status

- 29/29 in-CI tests pass (lib unit tests + 8 synthetic translator smoke tests + 4 transformer-prep + 7 shape-ops + 6 fusion/cache/runtime smoke + 4 transformer-prep)
- 18 `#[ignore]`-marked real-model tests load external `.onnx`/`.nnef.tgz` files
- clippy clean for tract-coreml
- No CPU-path regression (tract-core 237/237 unchanged)

### Phase summaries

- **Phase 2**: 6/6 gates green. MobileNet 533 → 2 CoremlOps with 185 mW ANE sustained. See [`notes/phase-2-closure.md`](../../notes/phase-2-closure.md).
- **Phase 3**: 9/9 gates green. 12 new op-translator families, persistent compile cache, convex-region union, multi-node InstanceNorm fold. Canaries: MobileNet, SqueezeNet, MODNet. See [`notes/phase-3-closure.md`](../../notes/phase-3-closure.md).
- **Phase 4 (in-flight)**:
  - **RVM** as 4th canary closed. 100% op coverage; ANE didn't engage (recurrent state shapes; needs investigation).
  - **Transformer-prep batch**: Softmax + general MatMul + LayerNorm fold + RmsNorm standalone (~1,000 LoC).
  - **SAM 2 Hiera-Tiny encoder** as 5th canary (transformer-class). Required Reshape + MoveAxis + Pad translators (defrag pass) and major general_matmul evolution (v3 batch-reorder + v4/v5/v6 external-shape boundary). 47→39 CoremlOps; all 81 EinSums absorbed. Perf 4.40 → 1.62–2.00 s (2.2–2.7× speedup). ANE compile fails at the current fragmentation level. See [`notes/phase-4-sam2-closure.md`](../../notes/phase-4-sam2-closure.md).
  - **DFN3 decoder** as 6th canary (first non-ONNX, NNEF format). RTF 0.066 (15× faster than real-time) at 1s audio chunks. 3 CoremlOps for the OUTSIDE-scan compute; 2 GRU scan loops on CPU (no MIL scan equivalent). See [`notes/phase-4-dfn3-closure.md`](../../notes/phase-4-dfn3-closure.md).
- **Upstream feedback log**: [`notes/tract-upstream-feedback.md`](../../notes/tract-upstream-feedback.md) — running list of tract issues hit during this work, with priority + suggested fixes. ~12 of 21 entries benefit users beyond this backend (CPU users + every accelerator backend).

## How it works

1. **Tract pipeline integration**. `CoremlRuntime` impls `tract_core::Runtime`.
   `prepare_with_options(model)` runs `CoremlTransform`, then the standard
   `into_optimized()` + `TypedSimplePlan::build`. The transform is also
   registered via the `inventory`-based factory so it can be looked up by name.

2. **Subgraph identification**. `fusion::identify_subgraphs` walks the typed
   model and partitions translatable nodes into maximal connected subgraphs
   via a union-find with two cycle-avoiding rules:
   - **All-or-nothing**: a node only joins a subgraph when *all* its non-const
     data inputs are translatable.
   - **Convex-region**: when a node's translatable predecessors sit in
     different existing subgraphs, the union is allowed iff the merged
     region's quotient graph is acyclic — checked by a forward BFS from the
     candidate-merger set through CPU nodes; any path that loops back into
     the set indicates a cycle and rejects the merger. This is a relaxation
     of the original "single-root" rule that lets us collapse MODNet from
     9 → 2 CoremlOps.

3. **MLPackage materialisation**. `fusion::build_subgraph_mlpackage` walks the
   subgraph in topological order, dispatches each node to its per-op
   translator (`ops::conv::analyse_conv` + `emit_conv_mil`,
   `ops::binop::analyse_binop` + `emit_binop_mil`,
   `ops::cast::analyse_cast` + `emit_cast_mil`,
   `ops::einsum::analyse_einsum` reusing `emit_conv_mil`), names every value
   uniquely within the program, absorbs const-fed inputs into the MILBlob v2
   weight file, and writes the resulting MIL Program + weight blob as a
   `.mlpackage` directory.

4. **Deferred-retry transform walk**. `transform::CoremlTransform` walks the
   source model in topo order; for each node, either wires it (or its
   `CoremlOp` if it's a subgraph root) when all required inputs are mapped, or
   defers to the next pass. Necessary because subgraph members are not
   topologically contiguous in the source — non-translatable nodes can sit
   between two members of the same subgraph.

5. **Per-subgraph CoremlOp**. Each subgraph is replaced in the typed model with
   a single `CoremlOp` node carrying:
   - an `Arc<CoremlContext>` (the loaded `MLModel`)
   - positional CoreML feature names for inputs / outputs
   - per-input + per-output MLPackage shape (so `eval` can reshape between
     tract-side and MLPackage-side ranks — Conv on CHW data prepends `N=1`
     internally)
   - tract `output_facts` (so `TypedOp::output_facts` answers without
     querying the MLModel)

6. **Inference**. `CoremlOp::eval` converts each input tract `Tensor` to an
   `MLMultiArray` (allocates with the MLPackage's shape; flat memcpy when
   strides are contiguous, stride-walked otherwise — Core ML pads inner dims
   for ANE alignment), wraps in `MLDictionaryFeatureProvider`, calls
   `MLModel::predictionFromFeatures_error`, and reads each named output back
   into a contiguous tract `Tensor` via stride-walked copy.

## Op coverage (Phase 3 in-flight)

| tract op | MIL primitive | Status | Notes |
|---|---|---|---|
| `Conv` (2D, NCHW or CHW, OIHW kernel, groups ≥ 1, F16 in/out + F16 const weight + F16 const bias, Explicit/Valid padding) | `mb.conv` | ✓ Phase 1 | Cast(Const) chains for weight/bias are walked; scalar bias broadcast to `[out_channels]` |
| `TypedBinOp(Add \| Mul \| Max \| Min \| Sub)` (F16) | `mb.add` / `mb.mul` / `mb.maximum` / `mb.minimum` / `mb.sub` | ✓ Phase 2 + Phase 3 ext | NumPy-style broadcasting (rank-4 normalised); const inputs absorbed into the MILBlob v2 weight file; ReLU lower-bound surfaces as `Max(x, 0_const)` and rides the const-absorption path for free; Min/Sub added in Phase 3 to cover Clip's upper bound + InstanceNorm's `x - mean` step |
| `Cast` (any of F16 / F32 / I32 / I8) | `mb.cast` | ✓ Phase 2 | Rank-preserving; Cast(Const) chains are absorbed by `const_tensor` upstream rather than emitted |
| `EinSum` 1×1 conv patterns (axes `IHW,OI->OHW`, `NIHW,OI->OHW`, `IHW,OI->NOHW`, `NIHW,OI->NOHW`, `NIHW,I->NOHW`, F16) | `mb.conv` | ✓ Phase 2 + Phase 3 ext | Reshapes the `[O,I]` weight to `[O,I,1,1]` and reuses `conv::emit_conv_mil`. The `->NOHW` variants (Phase 3) cover 1×1 convs whose output feeds a Concat. The rank-1-weight variant `NIHW,I->NOHW` (Phase 3.H) covers the single-output-channel case (e.g. MODNet's final mask projection) — weight `[I]` reshaped to `[1, I, 1, 1]` |
| `TypedConcat` (F16, all inputs same rank 2..=4) | `mb.concat` | ✓ Phase 3 | Variadic input via `arg_names` helper; rank-4 padding (axis shifted); `interleave=false` |
| `MaxPool` 2D (F16, NCHW or CHW, kernel rank 2, Explicit/Valid padding, no `with_index_outputs`) | `mb.max_pool` | ✓ Phase 3 | Same `PoolSpec` machinery as Conv; `pad_type` ∈ {"valid", "custom"}; `ceil_mode=false` |
| `Reduce<Sum \| Max \| Min \| Prod \| MeanOfSquares>` (F16, rank 2..=5, sorted axes) | `mb.reduce_sum` / `_max` / `_min` / `_prod` / `square + reduce_mean` | ✓ Phase 3 | `keep_dims=true` (matches tract's reduce-keeps-rank semantics); `MeanOfSquares` is lowered to two MIL ops since there's no direct equivalent; `ArgMax/ArgMin` deferred |
| `AxisOp::Rm(axis)` (F16, axis size 1, output rank ≥ 1) | `mb.squeeze` | ✓ Phase 3 | Rank-changing — preserves natural rank end-to-end (no rank-4 padding) so the output rank flows correctly to downstream tract consumers |
| `AxisOp::Add(axis)` (F16, output rank ≤ 5) | `mb.expand_dims` | ✓ Phase 3 | Insert size-1 axis at position; complement to `RmAxis` |
| `ElementWiseOp` (F16): `Sigmoid`, `Tanh`, `Silu`, `HardSwish`, `GeluApproximate`, `Gelu`, `Erf`, `Rsqrt`, `Sqrt` | `mb.sigmoid` / `tanh` / `silu` / `hard_swish` / `gelu` / `erf` / `rsqrt` / `sqrt` | ✓ Phase 3 | Single dispatcher via `ElementWiseMiniOp.name()` string match; pointwise so output shape == input shape; GeLU emits `mode="TANH_APPROXIMATION"` for the approximate variant; Rsqrt/Sqrt require explicit MIL `epsilon` const (we emit `1e-12`) |
| `Resize` (from `tract-onnx-opl`, F16): `Linear` / `Nearest` interpolator, F16 input rank 3 (CHW) or 4 (NCHW), const scales OR const sizes | `mb.resize_bilinear` / `mb.resize_nearest_neighbor` | ✓ Phase 3 | `HalfPixel` / `PytorchHalfPixel` → `DEFAULT`, `AlignCorners` → `STRICT_ALIGN_CORNERS`, `Asymmetric` → `OFFSET_CORNERS`. Empty (length-0) ROI input allowed (the typical ONNX 4-input layout); non-empty ROI rejected. Cubic interpolation deferred (no direct MIL equivalent) |
| `Slice` (single-axis `start..end`, F16, rank 2..=5, concrete bounds) | `mb.slice_by_index` | ✓ Phase 3 | Per-axis begin/end/stride vectors (rank-length); `begin_mask` / `end_mask` / `squeeze_mask` are **bool vectors** (NOT int32 bitmasks as legacy MIL docs suggest). All non-sliced axes get mask=true to use full range; rank preserved (`squeeze_mask` always all-false) |
| `EinSum` matmul-shaped (`mkba,kn->n`, `k,kn->mnab`, F16) | `reshape` + `mb.matmul` + `reshape` | ✓ Phase 3 | The dense classifier-head pattern tract emits when not folded into 1×1 conv. Reshape input to `[1, K]`, matmul with weight `[K, N]`, reshape output to tract's expected rank. Const weight absorbed into the MILBlob v2 weight file |
| **Multi-node InstanceNorm chain** (ReduceSum + Mul(1/N) + Sub + Reduce<MoS> + Add(eps) + Rsqrt + final Mul, F16, NCHW) | `mb.instance_norm` | ✓ Phase 3 | Pattern detector anchored at the final Mul; verifies the full chain and that no intermediate has external consumers. Emits one `mb.instance_norm` (gamma=1, beta=0; per-channel scale/bias come via separate BatchNorm ops). Empirically perf-neutral on M1 Pro (Apple's MIL compiler was already fusing the decomposition optimally) — kept for cleaner MLPackages and faster Apple compile time |

Anything else falls back to CPU. The next op categories in priority order: **ConvTranspose** (1D + 2D, decoder + DFN3), **MatMul / Gemm**, **Softmax**, **Reshape / Transpose**, **InstanceNorm / LayerNorm / RMSNorm** (direct MIL ops vs the decomposed Sub/Mul/Rsqrt/Reduce path we currently support), **AveragePool**, **Embedding / Gather**. See [`../../notes/phase-3-progress.md`](../../notes/phase-3-progress.md) §5 for sequencing.

## Architecture rationale (for upstream reviewers)

Fourteen design choices, with the underlying analysis:

1. **No `tract-gpu` dep** — Core ML's MLMultiArray ownership and per-prediction
   memory model don't fit the gpu shared abstractions. See `notes/phase-0-recon.md` §7.4.
2. **Pure-Rust runtime MIL build via vendored `.proto`** — avoids a Python
   dependency for tract users; ~50× faster per build than subprocess to
   coremltools. See `notes/phase-1-spike.md`.
3. **Subgraph as the dispatch unit** — Apple's scheduler decides ANE/GPU/CPU
   per MLPackage, so per-op MLPackages can't engage ANE. Empirically validated
   in Phase 2: 47 small MLPackages = 24 mW ANE; 2 large MLPackages (the
   EinSum unlock) = 185 mW ANE. See `notes/phase-1-mvp.md` §4 and
   `notes/phase-2-closure.md` §2.
4. **MLProgram only, no NeuralNetwork format** — narrower MLProgram op coverage
   targeting; ANE friendliness; no `scaled_dot_product_attention` in
   NeuralNetwork. See `references/ort-coreml-architecture.md`.
5. **Vendored `.proto` at pinned commit** (`apple/coremltools` 9.0,
   `428d4b2`) — build determinism, no submodule complexity. See
   [`proto/MIL_PROTO_VERSION.md`](proto/MIL_PROTO_VERSION.md).
6. **`protox` instead of system `protoc`** — pure-Rust proto compilation; no
   `brew install protobuf` for tract-coreml users.
7. **Cycle-avoiding "all-or-nothing" union rule** — prevents the convex-region
   violation observed in MobileNet residual chains (`Add[i].inputs =
   (bn[i]_chain, Add[i-1].out)` would otherwise fuse `Add[i]` into a subgraph
   whose external_input depends on the subgraph's own output). See
   `notes/phase-2-progress.md` §1.1.
8. **Cycle-avoiding "single-root" union rule** — even with all-or-nothing,
   when a translatable node N's translatable predecessors P1 and P2 sit in
   *distinct* existing subgraphs S1 and S2, unioning through N would merge
   S1 ∪ S2 ∪ {N} whose external_inputs may transitively depend on each other
   through CPU chains. Required for MobileNet's cross-block residuals. See
   `notes/phase-2-closure.md` §4.9.
9. **EinSum-as-Conv targeted translation** — tract lowers 1×1 convolutions
   to `EinSum("IHW,OI->OHW", data, weight)` post-`into_decluttered`. Rather
   than write a generic einsum-to-MIL translator, we pattern-match these axes
   signatures and reuse `conv::emit_conv_mil` with the weight reshaped to
   `[O,I,1,1]`. This is the unlock that collapses MobileNet's 18 Convs +
   105 BN/ReLU/elemwise ops into 2 CoremlOps and engages ANE end-to-end.
   Generic einsum is Phase 3. See `notes/phase-2-closure.md` §4.10.
10. **Const absorption via the MILBlob v2 weight file** — BN's per-channel
    `Mul(data, scale_const)` and `Add(data, bias_const)` (and ReLU's
    `Max(data, 0_const)`) absorb the const input into the MLPackage's blob
    rather than emitting an immediate value. One mechanism for absorbed
    consts; tiny disk cost. See `notes/phase-2-closure.md` §4.11.
11. **Stride-aware `MLMultiArray` reads** — Core ML pads inner dims for ANE
    alignment (observed: a logical `[1, 8, 8, 8]` FP16 tensor came back with
    strides `[2048, 256, 32, 1]`). Naive `slice::from_raw_parts` returns
    garbage past the first row. See `feedback_mlmultiarray_strides_observed`
    in the project memory.
12. **EinSum NOHW pattern recognition** (Phase 3) — extends decision #9 to
    cover `IHW,OI->NOHW` and `NIHW,OI->NOHW`, the rank-4-output variants tract
    emits when a 1×1 conv feeds a Concat (Concat needs matching rank, so tract
    synthesises the explicit N=1 in the output). Without this, every
    fire-module concat boundary in SqueezeNet fragmented the graph into
    separate MLPackages. See `notes/phase-3-progress.md` §4.12.
13. **RmAxis preserves natural rank end-to-end** (Phase 3) — other translators
    pad MLPackage-side input to rank 4 because Conv/BinOp/Cast/MaxPool's
    natural CHW (rank 3) and NCHW (rank 4) shapes both fit cleanly there.
    Squeeze *changes* rank, so this padding breaks down. RmAxis preserves
    natural rank on both sides; works because RmAxis at model boundaries
    (e.g. SqueezeNet's classifier head) has no in-subgraph consumers.
    See `notes/phase-3-progress.md` §4.13.
14. **MLPackage size alone doesn't engage ANE** (Phase 3) — SqueezeNet at
    Phase 3 close has the same single-MLPackage shape as MobileNet at Phase 2
    close, but ANE Power is 0 mW vs MobileNet's 185 mW. Difference: total
    compute (SqueezeNet ~700 µs/inf vs MobileNet ~3,900 µs/inf). Apple's
    scheduler picks ANE only when per-call compute outweighs ANE submission
    overhead. **Subgraph fusion is necessary but not sufficient** — sharpens
    decision #3. See `notes/phase-3-progress.md` §4.14.

## Reproduction

```sh
# Library + integration tests (14 in-CI):
cargo test -p tract-coreml --tests

# Real-model tests (need ~/coding/dfn3-wasm-relaxed-simd/models/{mobilenetv2-7,squeezenet}.onnx):
cargo test -p tract-coreml --test mobilenet_e2e -- --ignored --nocapture
cargo test -p tract-coreml --test squeezenet_e2e -- --ignored --nocapture

# Perf bench (release, n=200, warmup=5) — expect ~125× on MobileNet, ~35× on SqueezeNet (M1 Pro):
cargo test -p tract-coreml --release --test mobilenet_e2e -- \
    --ignored --nocapture mobilenet_v2_perf_cpu_vs_coreml
cargo test -p tract-coreml --release --test squeezenet_e2e -- \
    --ignored --nocapture squeezenet_perf_cpu_vs_coreml

# powermetrics ANE measurement during sustained inference (two shells).
# MobileNet: expect ANE Power 180+ mW sustained on M1 Pro.
# SqueezeNet: expect ANE 0 mW (model too small — Apple's scheduler picks CPU).
cargo test -p tract-coreml --release --test mobilenet_e2e -- \
    --ignored --nocapture mobilenet_v2_sustained_for_powermetrics &
sudo powermetrics -A --show-extra-power-info -n 10 -i 1500 \
    | grep -E "(CPU|GPU|ANE) Power"

# Subgraph diagnostics (per-Conv eligibility + skip reasons):
TRACT_COREML_DEBUG_SUBGRAPHS=1 cargo test -p tract-coreml \
    --test mobilenet_e2e -- --ignored --nocapture mobilenet_v2_cpu_vs_coreml_correctness

# Hygiene (Phase 3 close):
cargo fmt -p tract-coreml --check
cargo clippy -p tract-coreml --tests --no-deps -- -D warnings
```

## Layout

```
tract/coreml/
├── Cargo.toml             # MIT-OR-Apache-2.0; objc2-core-ml 0.3, prost 0.14
├── README.md              # this file
├── build.rs               # protox + prost-build, disable_comments(["."])
├── proto/                 # 33 vendored .proto + MIL_PROTO_VERSION.md
├── src/
│   ├── lib.rs             # CoremlRuntime + register_runtime!
│   ├── compile_cache.rs   # persistent compile cache: ~/Library/Caches/tract-coreml/v1/<sha256>.mlmodelc/
│   ├── proto/             # prost-generated re-export tree
│   ├── context.rs         # CoremlContext (MLModel handle, mutex-protected)
│   ├── tensor.rs          # tract Tensor ↔ MLMultiArray, stride-aware
│   ├── coreml_op.rs       # CoremlOp impl Op + EvalOp + TypedOp
│   ├── mil/               # MIL builder primitives (value, op, blob, program)
│   ├── mlpackage.rs       # .mlpackage dir + Manifest.json writer
│   ├── ops/               # per-tract-op translators (conv, binop, cast, einsum, concat, maxpool, reduce, rm_axis, add_axis, activation, resize, slice, matmul, instance_norm, avgpool, broadcast) + shared helpers (const_tensor, shape_to_concrete_i64)
│   ├── transform.rs       # CoremlTransform — deferred-retry walk
│   └── fusion.rs          # Subgraph identification + multi-op MLPackage
└── tests/
    ├── compile_cache_smoke.rs   # cache misses then hits on second transform; warm load < cold/2
    ├── coreml_op_smoke.rs       # single-op end-to-end (zero input + ones pattern)
    ├── runtime_smoke.rs         # Runtime API + transform-replaces-Conv check
    ├── fusion_smoke.rs          # Conv → Add(self, self) fuses into 1 CoremlOp
    ├── resize_smoke.rs          # Conv → Resize(2×, bilinear/nearest) fuses into 1 CoremlOp + runs end-to-end (Phase 3)
    ├── mobilenet_e2e.rs         # MobileNet v2: correctness + perf + sustained powermetrics
    ├── squeezenet_e2e.rs        # SqueezeNet 1.x: coverage probe + perf + sustained powermetrics (Phase 3 canary)
    ├── modnet_e2e.rs            # MODNet (segmentation): coverage probe + perf (CoreML-only — tract Resize is F32-only) (Phase 3 canary)
    ├── mediapipe_selfie_e2e.rs  # MediaPipe Selfie probe — currently blocked by tract-onnx symbolic-batch-size handling (Phase 3 deferred)
    └── rvm_e2e.rs               # RVM (Robust Video Matting): coverage probe + perf + sustained powermetrics (Phase 4 canary)
```

## Dependencies

| Crate | Version | Why |
|---|---|---|
| `tract-core` | workspace | the `Runtime` / `ModelTransform` / typed model APIs |
| `objc2-core-ml` | 0.3 | Rust bindings to `MLModel`, `MLMultiArray`, `MLPredictionOptions`, `MLComputeUnits` |
| `objc2-foundation` | 0.3 | `NSURL`, `NSString`, `NSDictionary`, `NSArray`, `NSNumber` |
| `objc2` | 0.6 | Obj-C runtime support |
| `prost` | 0.14 | runtime protobuf encoding (matches workspace) |
| `half` | workspace | `f16` for FP16 weights / activations |
| `serde_json` | workspace | `Manifest.json` writer |
| `uuid` | 1 | UUIDs in `Manifest.json` |
| `prost-build` (build) | 0.14 | proto codegen |
| `protox` (build) | 0.9 | pure-Rust proto compiler — avoids system `protoc` |

No `tract-gpu`, no Python, no `protoc` system binary.

## License

MIT OR Apache-2.0 — matches the rest of the tract workspace.
