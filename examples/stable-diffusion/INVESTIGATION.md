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
