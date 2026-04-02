# Recipe: factorizing a GPU op between CUDA and Metal

This describes the pattern used to factorize the Reduce op. Follow the same
steps for other ops (binary, softmax, etc.).

## Before

Each backend had its own copy of:
- The op enum (e.g. `Reducer` in both `cuda/src/kernels/nn/reduce.rs` and `metal/src/kernels/nn/reduce.rs`)
- The op wrapper struct (e.g. `CudaReduce` / `MetalReduce`) with identical Op/EvalOp/TypedOp impls
- A `from_tract_core()` that maps core ops to the backend op

The only real differences were:
- How to launch the kernel (cudarc vs Metal command buffers)

## After

### 1. Shared enum in `gpu/src/ops/`

Move the op enum (e.g. `Reducer`) into `gpu/src/ops/reduce.rs` with:
- Variant definitions
- `Display` impl (kernel name fragment — e.g. "sum", "prod", "mean_of_squares")
- `is_supported_dt()`, `is_logic()`, and other predicate methods
- `from_tract_core()` mapping from `core::ops::nn::Reducer`

### 2. Shared op struct with fn pointer for dispatch

`GpuReduce` in `gpu/src/ops/reduce.rs`:
- Stores: `axes`, `reducer`, `backend_name: &'static str`, `dispatch: DispatchReduceFn`
- `DispatchReduceFn = fn(&Reducer, &DeviceTensor, usize, &DeviceTensor) -> TractResult<()>`
- Implements `Op`, `EvalOp`, `TypedOp` once — shared across backends
- `eval_with_session` calls `(self.dispatch)(...)` directly
- `PartialEq`/`Hash` are manual impls that ignore the fn pointer (compare by axes + reducer + backend_name)

### 3. Stream access

The `gpu` crate has no stream concept. Each backend owns its stream in its
own thread-local:
- CUDA: `cuda/src/context.rs` has `CUDA_STREAM` TLS and `with_cuda_stream()`
- Metal: `metal/src/context.rs` has `METAL_STREAM` TLS and `with_metal_stream()`

Dispatch functions access the stream internally — they do not receive it
as a parameter.

### 4. Backend kernel launch function

In `cuda/src/kernels/nn/reduce.rs`, the launch function accesses the
stream via its own TLS:

```rust
pub fn cuda_reduce_launch(
    reducer: &Reducer,
    input: &DeviceTensor,
    axis: usize,
    output: &DeviceTensor,
) -> TractResult<()> {
    crate::with_cuda_stream(|stream| {
        // ... kernel launch code
    })
}
```

### 5. Wiring in transform.rs

No per-op wrapper module needed. Transform calls `GpuReduce::from_tract_core`
directly with the backend's launch function:

```rust
// in can_translate_to_cuda_op:
GpuReduce::from_tract_core(op, "Cuda", cuda_reduce_launch)
    .is_ok_and(|op| op.reducer.is_supported_dt(input_dts[0]))

// in translate_node:
Box::new(GpuReduce::from_tract_core(op, "Cuda", cuda_reduce_launch)?)
```

## Checklist for the next op

1. Move enum + predicates + `from_tract_core` + `Display` to `gpu/src/ops/`
2. Create `GpuXxx` struct with fn pointer dispatch, impl Op/EvalOp/TypedOp
3. Define `DispatchXxxFn` type alias (no stream parameter)
4. In each backend kernel file: make launch fn access its own stream TLS internally
5. In each backend transform.rs: call `GpuXxx::from_tract_core(op, "Backend", launch_fn)`
6. Delete the backend `ops/xxx.rs` wrapper
