# Stable Diffusion ONNX Dynamic Batch — Investigation Log

## Goal
Export SD 1.5 UNet (and VAE decoder) with dynamic batch axis via torch dynamo ONNX export, then load and run in tract with concrete batch sizes.

## Context
- SD 1.5 pipeline: text encoder, UNet, VAE decoder
- For classifier-free guidance, UNet needs batch=2N (N uncond + N cond)
- Single image generation already works (batch=1, two separate UNet calls)
- Want batched generation: one UNet call per step with batch=2N

## Attempts So Far

### 1. Dynamo export with dynamic_axes (default torch.onnx.export)
- Export: `sample=(2,4,64,64)`, `timestep=[999,999]`, `encoder_hidden_states=(2,77,768)`, all with `dynamic_axes={...: {0: "batch"}}`
- Result: ONNX graph has `batch` symbol on all three inputs
- **Problem in tract**: Internal value_info tensors carry the `batch` symbol. When we set concrete input facts (e.g. `2,4,64,64,f32`), tract tries to unify `Sym(batch)` with `Val(2)` and fails at Unsqueeze/Reshape nodes.
- Tried `--onnx-ignore-output-shapes`: still fails (same unification error)
- Tried `--onnx-ignore-output-shapes --onnx-ignore-value-info`: different error — `Reshaping [1, 64, 64, 320] to [batch, 4096, 320]` Invalid. A constant intermediate tensor has shape frozen to batch=1 from the trace.

### 2. Legacy export (dynamo=False) with dynamic_axes
- Export: same inputs but with `dynamo=False`
- Result: Proper dynamic batch on sample/encoder_hidden_states, timestep stays `1,i64`
- **tract loads and runs fine** with `--set batch=N` or concrete input facts
- **Problem**: Legacy-exported model is much larger (more ops), causes CUDA OOM on 32GB GPU
- CPU works but slow

### 3. ONNX → NNEF conversion
- Tried `tract dump --nnef` to pre-bake batch size into NNEF
- Text encoder: symbol unification error on dim 77
- UNet/VAE: `Resize` (nearest upsample) has no NNEF serializer, even with `--opl`
- With `-O`: `OptMatMulPack` has no serializer

## Key Issues to Investigate
1. Why does dynamo export bake intermediate shapes as constants?
2. Can tract's ONNX loader be made to handle the `batch` symbol from dynamo?
3. Is there a way to export with dynamo that produces truly dynamic intermediate shapes?
4. Alternative: can we add `ignore_value_info`/`ignore_output_shapes` to the public tract API?

## Investigation

### Entry 1: Symbolic batch survives the full pipeline

Starting from dynamo export with `dynamic_axes` on all inputs (sample, timestep, encoder_hidden_states), all with `{0: "batch"}`.

Tested each tract pass in sequence **without overriding input facts** (keeping symbolic `batch`):
- `--pass load` — OK. Inputs: `batch,4,64,64,F32` / `batch,I64` / `batch,77,768,F32`
- `--pass analyse` — OK. Shapes propagate symbolically.
- `--pass incorporate` — OK.
- `--pass type` — OK. Node count grows (4650+) as inference ops get translated.
- `--pass declutter` — OK. Node count drops (3134). Silu detected.
- `--pass optimize` — OK. Model fully optimized with symbolic `batch`.

**Key finding**: tract handles symbolic `batch` through the entire pipeline — the issue in previous attempts was from trying to concretize batch via `-i 2,4,64,64,f32` which conflicted with `batch` symbol in internal value_info.

### Entry 2: CLI run with --cuda works, Rust API OOMs

**CLI commands that work:**
```
tract assets/unet.onnx -O run --set batch=2 --allow-random-input          # CPU, OK
tract assets/unet.onnx -O --cuda run --set batch=2 --allow-random-input   # CUDA, OK, 24s
tract assets/unet.onnx --cuda run --set batch=2 --allow-random-input      # CUDA no -O, OK, 24s
```

**Key finding on -O vs --cuda** (from `cli/src/params.rs:1005-1013`):
- `--cuda` supersedes `-O` for runtime selection
- With `--cuda`: runtime = `"cuda"`, model goes to `runtime.prepare_with_options(typed_model)` directly
- With `-O` alone: runtime = `"default"`, which calls `into_optimized()` inside `DefaultRuntime::prepare()`
- Without either: runtime = `"unoptimized"`
- The CUDA runtime handles optimization internally in its `prepare()`.

**Rust API OOM:**
- `gpu.prepare(onnx.load("unet.onnx")?.into_model()?)` → process killed (OOM, 137)
- This should be equivalent to what the CLI does with `--cuda` (no `-O`)
- Same machine, 32GB GPU, 32GB free before run
- The CLI with identical flow works. Something differs between the public API path and the CLI path.

**Not the issue:**
- `ConcretizeSymbols` — removed, same OOM
- Missing `-O` — CLI works without `-O` too

### Entry 3: --readings reveals CUDA prepare eats all RAM

**CPU path (`-O`, no CUDA)** — from `readings.out`:
| Event | RSS |
|---|---|
| proto_model_loaded | 16 MB |
| model_loaded | 3.5 GB |
| after type | 3.5 GB |
| after declutter | 3.7 GB |
| model_ready (optimized) | 4.2 GB |

Healthy: 4.2GB peak for a 3.3GB weights model.

**CUDA path (`--cuda`, no `-O`)** — from `readings.out`:
| Event | RSS |
|---|---|
| model_loaded | 3.5 GB |
| after type | 3.5 GB |
| after declutter | 3.8 GB |
| after before-optimize | 3.8 GB |
| (then CUDA prepare starts...) | |
| last reading before kill | **60 GB** (and climbing) |

VSZ reached 77GB. 115 billion total allocations. Process killed at 60GB RSS.

**Key observation**: The CUDA runtime worked previously with the dynamo export for batch=1 (non-dynamic). The difference is now the model has symbolic `batch` dimension. The CUDA prepare with symbolic shapes triggers pathological memory allocation.

**Previous working run**: was with a model exported via dynamo without dynamic_axes (fixed batch=1), where `--cuda` worked fine in 5s and ~4GB.

### Entry 4: `-vv` reveals PropConst is the culprit

Running with `-vv` shows the last output before OOM kill:

```
[DEBUG tract_core::optim] applying patch #255: PropConst(4213)/0    # fast, microseconds
[DEBUG tract_core::optim] applying patch #0: PropConst(95)/0        # second round starts
[DEBUG tract_core::optim] applying patch #1: PropConst(112)/0       # ~600ms each
...
[DEBUG tract_core::optim] applying patch #9: PropConst(212)/0       # ~2.4s each, growing
[DEBUG tract_core::optim] applying patch #10: PropConst(219)/0      # ~2.4s
[DEBUG tract_core::optim] applying patch #11: PropConst(235)/0      # ~2.4s
...
[DEBUG tract_core::optim] applying patch #21: PropConst(415)/0      # still going, then killed
```

First round: 256 PropConst patches, each taking microseconds.
Second round: PropConst patches take ~600ms to 2.4s each and **growing**. Memory climbs from 3.8GB to 60GB during this phase.

**PropConst = constant propagation.** It's trying to evaluate/fold constants through the graph. With symbolic `batch` dimension, something causes the intermediate tensors to blow up in size during the second optimization round.

**Note**: This is happening in `tract_core::optim`, not in the CUDA runtime itself. The CUDA runtime's `prepare()` calls `into_optimized()` which runs the codegen optimizer, and PropConst is part of that optimizer pass.

### Entry 5: Root cause — MultiBroadcastTo materializes giant tensors

Instrumented `prop_const.rs` to log what each PropConst produces. The second optimization round reveals:

```
PropConst: node #112 "node_linear_3.broadcast" (MultiBroadcastTo) -> 4096,320,320,F32   # 1.6GB
PropConst: node #121 "node_linear_4.broadcast" (MultiBroadcastTo) -> 4096,320,320,F32   # 1.6GB
...
PropConst: node #212 "node_MatMul_274...broadcast" (MultiBroadcastTo) -> 4096,320,1280,F32  # 6.4GB
PropConst: node #219 "node_MatMul_274...broadcast" (MultiBroadcastTo) -> 4096,320,1280,F32  # 6.4GB
PropConst: node #235 "node_MatMul_284.broadcast" (MultiBroadcastTo) -> 4096,1280,320,F32  # 6.4GB
PropConst: node #244 "node_conv2d_4.einsum.broadcast" (MultiBroadcastTo) -> 64,64,320,320,F32  # 1.6GB
```

**The 4096 = 64×64** — the spatial resolution of the latent. These are attention weight matrices `(320,320)` being broadcast to `(H*W, 320, 320)`. Since the spatial dims are concrete (only `batch` is symbolic), PropConst considers these as foldable constants and materializes them.

Each `MultiBroadcastTo` creates a 1.6GB or 6.4GB dense tensor. With ~30 such patches, memory climbs to 50+GB before OOM kill.

**Why the first round doesn't have this:** The first round folds small constant tensors (biases, scales). After round 1, the optimizer creates new Einsum nodes that decompose matmul into broadcast+mul patterns, creating the `MultiBroadcastTo` nodes that round 2 then tries to fold.

**PropConst guard (line 43):** `fact.shape.volume().as_i64().is_some_and(|d| d < 1024)` — this only applies when the node's input has **multiple successors**. These `MultiBroadcastTo` nodes have single-successor chains, so the guard is bypassed entirely.

### Entry 6: The broadcast feeds into matmul — unnecessary materialization

After declutter, `node_linear_3` is an `EinSum` with formula `amk,kn->amn (F32)`:
- Input A: `batch,4096,320` (activation, from LayerNorm)
- Input B: `320,320` (const weight)
- Output: `batch,4096,320`

This is a standard batched matmul — the weight `(k,n)` doesn't need broadcasting to `(m,k,n)`.

During the **codegen optimizer** (CUDA path), the EinSum gets decomposed into broadcast+multiply ops, creating `MultiBroadcastTo(320,320 -> 4096,320,320)`. Then PropConst eagerly folds this broadcast since the weight const has a single successor chain.

**The broadcast is an artifact of the codegen decomposition.** The original EinSum handles the weight natively without materialization. The codegen shouldn't be broadcasting a `(k,n)` weight to `(m,k,n)` — it should emit a proper batched matmul kernel instead.

**Same pattern repeats** for every attention layer in the UNet (~30 times), each creating 1.6-6.4GB constants.

### Entry 7: Confirmed — broadcast is intended to stay logical, PropConst materializes it

The CUDA rewrite rule `add_broadcast_pre_matmul` (cuda/src/rewrite_rules/add_matmul_broadcast.rs) intentionally inserts `MultiBroadcastTo` before the GEMM. The intent is for the CUDA kernel to consume the weight via stride-0 broadcasting (reading the same `(320,320)` for every batch element) — a zero-cost logical reshape.

However, `MultiBroadcastTo::eval_with_session` (core/src/ops/array/broadcast.rs:28) calls `broadcast_to_shape` which does `ndarray::broadcast().into_owned()` — a **physical dense copy**.

PropConst sees: stateless op + constant input + single successor → evaluates it → materializes 1.6GB dense tensor.

**The bug:** PropConst runs after the CUDA rewrite rules in the optimizer. The rewrite inserts broadcasts meant to be consumed logically by kernels, but PropConst eagerly folds them into dense constants before the kernel ever sees them.

**Possible fixes:**
1. **PropConst size guard:** Add a volume limit to the single-successor path too (e.g. skip if output > 100MB)
2. **Mark the broadcast as non-foldable:** The CUDA rewrite could mark the MultiBroadcastTo so PropConst skips it
3. **Ordering:** Run PropConst before the CUDA rewrites, not after
4. **Lazy broadcast tensor:** Make `MultiBroadcastTo` produce a stride-based tensor that doesn't allocate

### Entry 8: PropConst size guard fix — works but 9x performance regression

**Fix (branch `fix-propconst-broadcast-oom`):** In `prop_const.rs`, skip folding when `output_mem > max(input_mem, 1MB)`. Uses `TypedFact::mem_size()` for input, `datum_type.size_of() * volume` for output. Applied to both the initial eval and the chained successor loop.

**Result:** No OOM. Model loads in ~4GB. But runtime goes from 24s to 3m39s (9x slower).

**Why slow:** Comparing CUDA op histograms with and without `--set batch=4`:

| Op | With --set | Without --set |
|---|---|---|
| CudaGgmlGemm | 323 | 323 |
| CudaMultiBroadcastTo | 234 | 234 |
| (everything else) | identical | identical |

Both graphs have identical op counts — all nodes are lowered to CUDA. The 234 `CudaMultiBroadcastTo` ops are the problem. Previously PropConst folded them into dense constants (zero runtime cost, huge memory). Now they execute on GPU every forward pass.

**Root cause of slowness:** Confirmed by nsight-systems profiling.

### Entry 9: nsys profiling — `copy_nd3_f32` is the bottleneck

Profiled `--set batch=1` on the left (eager) vs right (late) of `run`:

| | Eager (4s) | Late (3m35s) |
|---|---|---|
| **Top GPU kernel** | `cutlass_sgemm` 9.6% | `copy_nd3_f32` **78.7%** |
| `copy_nd3_f32` instances | 160 @ 6µs avg | 384 @ **4.3ms** avg |
| GEMM kernel | `cutlass_sgemm_128x128` (batched) | `ggml_matvec_ncols_1` (vector!) |
| GEMM instances | 80 cutlass + 64 others | 114+100+11 matvec |

**Key findings:**
1. `copy_nd3_f32` physically materializes the broadcast on GPU — 384 calls × 4.3ms = 1.6s of pure copy
2. The GEMM fell back from batched Cutlass SGEMM to `ggml_matvec` — a **scalar vector** kernel. The broadcast produces a batch of matrices but each has batch=1 spatial, so GGML treats it as individual matvecs
3. In eager mode, PropConst folds the broadcast at prepare time, so the GEMM receives a pre-expanded constant and Cutlass recognizes the full batch

**The real fix** should be:
- The CUDA GEMM kernel should handle stride-0 broadcasts natively without needing physical expansion
- Or `add_broadcast_pre_matmul` should not emit the broadcast when the spatial dims are being treated as batch — the original EinSum `amk,kn->amn` should lower directly to a batched GEMM with the weight shared across `m`

## Summary: EinSum lowering to CUDA — the full picture

### The EinSum

After declutter, the SD 1.5 UNet attention layers produce EinSums like:

```
#103 node_linear_3 (EinSum) amk,kn->amn
  Input A: batch,4096,320,F32   (activation from LayerNorm)
  Input B: 320,320,F32          (constant weight)
  Output:  batch,4096,320,F32
```

This is `torch.nn.Linear` — a batched matmul where `a=batch` (symbolic), `m=4096` (spatial=64×64), `k=n=320` (features). The weight `(k,n)` is shared across both `a` and `m`.

### Lowering path: EinSum → PrefixMatMul → CUDA

1. **EinSum → PrefixMatMul**: The EinSum `amk,kn->amn` is lowered to `PrefixMatMul` with prefix dims `[a,m]` and matmul dims `[k,n]`. At this point, `(batch,4096)` are "batch" dims and `(320,320)` are the matmul.

2. **`add_broadcast_pre_matmul`** (cuda/src/rewrite_rules/add_matmul_broadcast.rs): Detects that input B `(320,320)` has no batch prefix while A has `(batch,4096)`. Inserts `MultiBroadcastTo` to expand B from `(320,320)` to `(batch,4096,320,320)`. This is meant to make B's shape match A's batch prefix so the CUDA GEMM receives uniformly-shaped inputs.

3. **CUDA GEMM selection**: With the broadcast materialized, the GEMM sees matching batch dims and selects a batched Cutlass SGEMM kernel.

### Case 1: Concrete batch (eager `--set batch=1` before CUDA)

```
EinSum amk,kn->amn
  ↓ lower to PrefixMatMul
PrefixMatMul [1,4096] + [320,320]
  ↓ add_broadcast_pre_matmul
MultiBroadcastTo(320,320 → 1,4096,320,320) + PrefixMatMul
  ↓ PropConst folds the broadcast (input=320×320=400KB, output=1×4096×320×320=1.6GB)
Const(1,4096,320,320) + CudaGgmlGemm
  → Cutlass batched SGEMM, 4s total
```

PropConst materializes the broadcast at prepare time. 1.6GB per attention layer, but with batch=1 and the constants already on GPU, the GEMM is fast. Peak ~4GB RAM.

### Case 2: Symbolic batch (late `--set batch=1` at runtime), WITHOUT PropConst guard

```
EinSum amk,kn->amn
  ↓ lower to PrefixMatMul
PrefixMatMul [batch,4096] + [320,320]
  ↓ add_broadcast_pre_matmul
MultiBroadcastTo(320,320 → batch,4096,320,320) + PrefixMatMul
  ↓ PropConst: output volume = batch×4096×320×320 — unknown at optimize time?
```

**Actually NO** — PropConst evaluates the op at optimize time. With symbolic batch, `MultiBroadcastTo::eval` needs concrete dims from `session.resolved_symbols` (line 27 of broadcast.rs). With no symbols resolved, it would fail with `TooEarly`. So PropConst skips it (line 93 catches the error). But at runtime with `--set batch=1`, the broadcast DOES execute — producing `(1,4096,320,320)` on GPU via `copy_nd3_f32` (4.3ms each), and the GEMM falls back to `ggml_matvec` (scalar). **OOM doesn't happen because batch=1 keeps things at 1.6GB per layer.**

Wait — but we DID see OOM earlier with symbolic batch. Let me reconsider: the MultiBroadcastTo shape is `batch,4096,320,320`. With symbolic `batch`, `eval` fails (`TooEarly`). But some of the other broadcasts DON'T involve `batch` — for example `64,64,320,320` (the conv einsum broadcasts). Those ARE concrete and PropConst folds them at 1.6GB each, ~30 times = 48GB → OOM.

**Corrected understanding**: The OOM comes from broadcasts where ALL dims are concrete (no `batch` involved), such as conv weight broadcasts `(320,320) → (64,64,320,320)`. The attention broadcasts `(320,320) → (batch,4096,320,320)` are actually safe because `batch` prevents evaluation.

### Case 3: Symbolic batch, WITH PropConst guard (the fix)

```
EinSum amk,kn->amn
  ↓ lower to PrefixMatMul
PrefixMatMul [batch,4096] + [320,320]
  ↓ add_broadcast_pre_matmul
MultiBroadcastTo(320,320 → batch,4096,320,320) + PrefixMatMul
  ↓ PropConst: skipped (output_mem > max(input_mem, 1MB))
CudaMultiBroadcastTo + CudaGgmlGemm
  → copy_nd3_f32 (4.3ms) + ggml_matvec (scalar), 3m35s total
```

No OOM, but 50x slower. The GEMM doesn't know how to batch over the broadcast.

### The core tension

- **PropConst folding is the "fast path"** — it materializes the broadcast once at prepare time, letting Cutlass see full batch dims for efficient SGEMM.
- **PropConst folding causes OOM** when the broadcast produces multi-GB tensors (especially conv weight broadcasts with concrete spatial dims).
- **Skipping the fold** is safe for memory but the CUDA runtime path (copy + matvec) is extremely slow.

### Possible resolutions

1. **Fold at prepare time, but on GPU**: Instead of PropConst materializing on CPU, have the CUDA runtime materialize the broadcast into GPU memory once during `prepare()`. This keeps the fast GEMM path without CPU RAM blowup.

2. **Make GEMM handle stride-0**: Teach `CudaGgmlGemm` to handle a weight with stride-0 in the batch dimension — shared across all batch elements. This is what cuBLAS `stridedBatchedGemm` with `strideB=0` does natively.

3. **Don't broadcast at all for `amk,kn->amn`**: The `add_broadcast_pre_matmul` rewrite could recognize that this pattern is just a standard batched matmul where the weight is naturally shared. Lower it directly to `stridedBatchedGemm(strideA=m*k, strideB=0, strideC=m*n)` without any broadcast node.

4. **Hybrid PropConst guard**: Allow folding if the output fits in a budget (e.g. 256MB), skip if it would blow up. This handles the common case while protecting against the pathological one. But it's fragile and model-dependent.

### Entry 10: EinSum lowering chains inventory

#### Common first step (core)

All paths start with the same two steps:

1. **`einsum_matmul::detect_all(model)`** — rewrites generic `EinSum` ops into `EinSumMatMul` ops by identifying `m`, `k`, `n` axes and "prefix" axes (all other axes = batch dims).

2. **`rewrite_einsum_to_prefix_matmul` rule** — rewrites `EinSumMatMul` into `PrefixMatMul` by:
   - Reordering A to `prefix+m+k` (or transposed `prefix+k+m`)
   - Reordering B to `prefix+k+n` (or transposed `prefix+n+k`)
   - Choosing least-cost transpose variant
   - Recording `transpose_a`, `transpose_b`, `transpose_c` flags

   **The prefix includes ALL axes not in {m,k,n}**, regardless of whether they appear in both inputs. For `amk,kn->amn`, prefix=`a`, so B is expected to have shape `a+k+n`. But B only has `k+n` — the `a` dim is missing.

#### CPU path (DefaultRuntime → `into_optimized`)

```
EinSum
  ↓ detect_all
EinSumMatMul {m,k,n, prefix="a"}
  ↓ rewrite_einsum_to_prefix_matmul(strict=true)
PrefixMatMul (A: a+m+k, B: a+k+n — B gets size-1 axis added for missing prefix)
  ↓ AsBlas: matmul_to_sgemm
SGemm (if not transposed, f32, no quant)
  ↓ codegen optimizer (PropConst etc.)
(optimized graph)
```

**CPU handles the missing prefix** by broadcasting B's size-1 prefix dim at runtime via ndarray broadcasting. No explicit MultiBroadcastTo is inserted. SGemm's eval handles this natively.

#### CUDA path (CudaTransform)

```
EinSum
  ↓ detect_all
EinSumMatMul {m,k,n, prefix="a"}
  ↓ rewrite_einsum_to_prefix_matmul(strict=false)       [phase 0]
PrefixMatMul (A: a+m+k, B: a+k+n — B has size-1 prefix)
  ↓ add_broadcast_pre_matmul                              [phase 1]
MultiBroadcastTo(B: k+n → a+m+k+n) + PrefixMatMul
  ↓ translate_model (CudaTransform)                       [phase 2]
CudaMultiBroadcastTo + CudaGgmlGemm
  ↓ codegen optimizer (PropConst, fuse_axis_op etc.)      [phase 3]
(optimized CUDA graph)
```

**The CUDA path inserts an explicit `MultiBroadcastTo`** because `CudaGgmlGemm` expects matching batch dims. This is where the problem occurs — PropConst folds the broadcast into a giant constant.

**Key difference:** `strict=true` (CPU) vs `strict=false` (CUDA). With `strict=false`, the PrefixMatMul has `operating_dt=Some(F32)` which allows more flexible kernel selection. But this doesn't affect the broadcast issue.

#### Metal path (MetalTransform)

Same structure as CUDA — calls `rewrite_einsum_to_prefix_matmul(strict=false)` then has its own rewrite rules.

#### The fold-into-m optimization target

The optimization we're implementing modifies **step 2** (`rewrite_einsum_to_prefix_matmul`) to detect prefix axes that only appear in A (not in B), and fold them into `m` via a reshape. This way B doesn't need a broadcast at all:

```
EinSumMatMul amk,kn->amn (prefix="a", a only in A)
  ↓ fold 'a' into m
Reshape A: (batch,4096,320) → (batch*4096, 320)
PrefixMatMul (A: (batch*4096)+k, B: k+n) — no prefix mismatch!
Reshape C: (batch*4096, 320) → (batch, 4096, 320)
```

This benefits **all backends** (CPU, CUDA, Metal) since it happens in core before any backend-specific transforms.
