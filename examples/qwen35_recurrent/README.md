# tract-qwen35-recurrent

An in-process iterative decoder for Qwen3.5 hybrid recurrent/attention models
exported as Tract NNEF. It feeds convolution, Gated DeltaNet recurrent, and KV
caches between token steps without reloading the graph.

The runner provides:

- EOS stopping (Qwen3.5 default token ID `2`)
- a configurable near-EOS margin for numerically unstable final tokens
- repeated-suffix termination after three identical suffixes
- finite-logit validation
- per-token latency and throughput JSON
- FP32 fixture conversion to the graph's declared input types

```sh
cargo run --release -- \
  /path/to/model.nnef.tgz \
  /path/to/decoder-inputs.npz \
  16669 2 0.1
```

Arguments are model, NPZ fixture, token cap, EOS token, and EOS logit margin.
The NNEF graph must expose `input_ids`, `position_ids`, then its recurrent/cache
inputs, and return logits followed by cache outputs in the same order.

## Verification

The repository's CUDA Gated DeltaNet and causal-convolution tests must pass
before using this runner. Exact transcript parity should be checked against the
source model; FP16, FP32, and Q4 can make different greedy decisions near tied
logits. The reference Qwen3.5 OCR work found that lossless BF16 parity requires
native BF16 arithmetic rather than output-only rounding.

Licensed under MIT or Apache-2.0.
