# WebGPU floating-point precision

The backend preserves a model's declared tensor types by default. An application
can opt into FP16 internal storage before preparing the model for the `wgpu`
runtime:

```rust,ignore
model.transform("wgpu_fp16_f32_accum")?;
let runnable = tract::runtime_for_name("wgpu")?.prepare(model)?;
```

`wgpu_fp16_f32_accum` widens operands before multiplication and accumulates in
FP32 in the direct convolution and blocked/split-K matrix kernels.
`wgpu_fp16` uses FP16 arithmetic in those kernels instead. Split-K convolution
partial sums are stored and merged in FP32 in both modes. Other operators retain
their own arithmetic policies; neither option guarantees FP16 arithmetic in
every shader.

Both transforms preserve input and output types and order. They convert ordinary
internal FP32 tensors and weights, without identifying particular models or
calibrating their accuracy. Explicitly returned recurrent states therefore keep
their public type. The selected accumulation policy is recorded in the model
and captured by its GPU operations, including fused epilogues. Preparing another
model does not change it. Model output accuracy must be validated by the caller.

The transform leaves the graph unchanged if the active device does not support
`shader-f16`. Initialize WebGPU asynchronously first in a browser, as for other
WebGPU preparation calls. These transforms target the `wgpu` runtime on browser
and native adapters; they do not set precision policy for CUDA, Metal or CPU.

For a model that already declares FP16 tensors, the preparation-time scopable
knob `TRACT_WGPU_F16_ACCUMULATE_F32` selects accumulation in the same fast kernel
families. It defaults to `true`; a precision transform's model setting takes
precedence. It does not lower FP32 tensors or override explicit integer types.
