# CUDA recurrent Gated DeltaNet and causal-convolution update

This contribution adds the two single-token state-update kernels needed by
hybrid recurrent/attention decoders such as Qwen3.5:

- `CudaGatedDeltaNetRecurrent`: normalized query/key recurrent update with an
  FP32 matrix state and FP16 query/key/value/output tensors.
- `CudaCausalConv1dUpdateOp`: depthwise single-token causal-convolution cache
  update for a four-tap kernel.

Both operators provide typed-fact validation and CUDA-vs-CPU unit tests. The
GDN kernel assigns one CUDA block per head and one thread per value column;
state writes are coalesced and race-free. The recurrent state remains FP32 to
avoid long-context drift.

Verification on an RTX 5080 Laptop GPU (SM120):

```sh
export CUDA_HOME=/path/to/cuda-13
export LD_LIBRARY_PATH=/path/to/cuda-13/lib64:$LD_LIBRARY_PATH
cargo test -p tract-cuda qwen35_recurrent_step_matches_cpu -- --nocapture
cargo test -p tract-cuda qwen35_conv_update_matches_cpu -- --nocapture
```

The initial implementation deliberately keeps shape constraints explicit:
GDN head width 128 and causal-convolution width 4. Generalizing these should be
a separate change with corresponding kernel coverage.
