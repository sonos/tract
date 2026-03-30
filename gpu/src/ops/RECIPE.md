# Recipe: factorizing a GPU op between CUDA and Metal

This describes the pattern used to factorize the Reduce op. Follow the same
steps for other ops (binary, softmax, etc.).

## Before

Each backend had its own copy of:
- The op enum (e.g. `Reducer` in both `cuda/src/kernels/nn/reduce.rs` and `metal/src/kernels/nn/reduce.rs`)
- The op wrapper struct (e.g. `CudaReduce` / `MetalReduce`) with identical Op/EvalOp/TypedOp impls
- A `from_tract_core()` that maps core ops to the backend op

The only real differences were:
- How to access the stream (`CUDA_STREAM.with` vs `with_borrowed_metal_stream`)
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
- `DispatchReduceFn = fn(&dyn GpuStream, &Reducer, &DeviceTensor, usize, &DeviceTensor) -> TractResult<()>`
- Implements `Op`, `EvalOp`, `TypedOp` once — shared across backends
- `eval_with_session` calls `gpu::with_stream()` (the global thread-local callback) then `(self.dispatch)(...)`
- `PartialEq`/`Hash` are manual impls that ignore the fn pointer (compare by axes + reducer + backend_name)

### 3. Global GPU stream access in `gpu/src/lib.rs`

```rust
pub trait GpuStream: Downcast {}

type WithStreamFn = fn(&mut dyn FnMut(&dyn GpuStream) -> TractResult<()>) -> TractResult<()>;

thread_local! {
    static GPU_WITH_STREAM: Cell<Option<WithStreamFn>> = ...;
}

pub fn register_stream(f: WithStreamFn) { ... }
pub fn with_stream<R>(f: impl FnMut(&dyn GpuStream) -> TractResult<R>) -> TractResult<R> { ... }
```

`with_stream` uses type-erased `FnMut` + `Option<R>` to bridge the generic return
type through the non-generic fn pointer.

### 4. Backend registration (once per backend)

In `cuda/src/lib.rs`:
```rust
impl GpuStream for TractCudaStream {}

fn cuda_with_stream(f: &mut dyn FnMut(&dyn GpuStream) -> TractResult<()>) -> TractResult<()> {
    CUDA_STREAM.with(|stream| f(stream as &dyn GpuStream))
}
```

Registration happens in the `CUDA_STREAM` thread_local initializer:
```rust
thread_local! {
    pub static CUDA_STREAM: TractCudaStream = {
        let stream = TractCudaStream::new().expect("...");
        tract_gpu::register_stream(crate::cuda_with_stream);
        stream
    };
}
```

### 5. Backend kernel launch function

In `cuda/src/kernels/nn/reduce.rs`, the launch function takes `&dyn GpuStream`
and downcasts:

```rust
pub fn cuda_reduce_launch(
    stream: &dyn GpuStream,
    reducer: &Reducer,
    input: &DeviceTensor,
    axis: usize,
    output: &DeviceTensor,
) -> TractResult<()> {
    let stream = stream.cuda()?;  // StreamExt downcast
    // ... kernel launch code (unchanged from before)
}
```

`StreamExt` is defined in `cuda/src/context.rs`:
```rust
pub trait StreamExt {
    fn cuda(&self) -> TractResult<&TractCudaStream>;
}
impl StreamExt for &dyn GpuStream {
    fn cuda(&self) -> TractResult<&TractCudaStream> {
        self.downcast_ref().context("Expected a cuda stream")
    }
}
```

### 6. Wiring in transform.rs

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
3. Define `DispatchXxxFn` type alias
4. In each backend kernel file: make launch fn take `&dyn GpuStream`, downcast with `StreamExt`
5. In each backend transform.rs: call `GpuXxx::from_tract_core(op, "Backend", launch_fn)`
6. Delete the backend `ops/xxx.rs` wrapper
