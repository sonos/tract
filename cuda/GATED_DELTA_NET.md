# GPU recurrent Gated DeltaNet and causal-convolution update

This contribution adds CUDA and Metal implementations of the two single-token state-update kernels needed by
hybrid recurrent/attention decoders such as Qwen3.5:

- `GpuGatedDeltaNetRecurrent`: shared normalized query/key recurrent operation with an
  FP32 matrix state and FP16 query/key/value/output tensors.
- backend kernels for the depthwise single-token causal-convolution cache
  update for a four-tap kernel.

The shared operation and output allocation live in `tract-gpu`; CUDA and Metal
own only validation and kernel dispatch. Both backends provide CPU-reference
unit tests. The CUDA GDN kernel assigns one block per head and one thread per value column;
state writes are coalesced and race-free. The recurrent state remains FP32 to
avoid long-context drift.

Verification on an RTX 5080 Laptop GPU (SM120):

```sh
export CUDA_HOME=/path/to/cuda-13
export LD_LIBRARY_PATH=/path/to/cuda-13/lib64:$LD_LIBRARY_PATH
cargo test -p tract-cuda qwen35_recurrent_step_matches_cpu -- --nocapture
cargo test -p tract-cuda qwen35_conv_update_matches_cpu -- --nocapture
cargo test -p tract-metal qwen35_recurrent_step_matches_cpu -- --nocapture
cargo test -p tract-metal qwen35_conv_update_matches_cpu -- --nocapture
```

The initial implementation deliberately keeps shape constraints explicit:
GDN head width 128 and causal-convolution width 4. Generalizing these should be
a separate change with corresponding kernel coverage.
