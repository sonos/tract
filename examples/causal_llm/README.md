# causal_llm

A causal language model inference example using tract. Loads an NNEF model and a HuggingFace tokenizer to run text completion, with optional OpenAI-compatible HTTP serving.

## Binaries

- **complete** — Generate text from a prompt on the command line.
- **serve** — OpenAI-compatible completion server on `http://0.0.0.0:3000/v1/completions`.
- **client** — Send completion requests to the server.

## Usage

You need a tokenizer (`tokenizer.json`) and an NNEF model (`.nnef.tgz`).

### Obtaining a model

```sh
# NNEF model (Qwen3-1.7B, 4-bit quantized)
wget https://s3.amazonaws.com/tract-ci-builds/tests/llm/541/Qwen--Qwen3-1.7B-q40ef16/Qwen--Qwen3-1.7B-q40ef16.nnef.tgz
# Tokenizer
wget https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/tokenizer.json
```

### Text completion

```sh
cargo run --bin complete --release -- \
    -t path/to/tokenizer.json \
    -m path/to/model.nnef.tgz \
    -n 50 \
    "The capital of France is"
```

### Completion server

```sh
cargo run --bin serve --release -- \
    -t path/to/tokenizer.json \
    -m path/to/model.nnef.tgz
```

Then query it:

```sh
cargo run --bin client --release -- "The capital of France is"
```

## Options

- `-n <N>` — Number of tokens to generate (default: 20)
- `--force-cpu` — Disable CUDA/Metal, run on CPU only

## Library

The `CausalLlmModel` struct can be used as a library. It handles KV cache detection, runtime selection (CUDA > Metal > CPU), and prompt chunking automatically.

```rust
let llm = CausalLlmModel::from_paths("tokenizer.json", "model.nnef.tgz")?;
let mut state = llm.spawn()?;
state.append_text("Hello world")?;
for _ in 0..20 {
    state.generate_next_token()?;
}
```

## Speculative decoding

A cheap *drafter* proposes several tokens; the target verifies them in one
forward pass and accepts the longest greedy-matching prefix plus its own next
token. Output is identical to plain greedy decoding (`generate_next_token`).

The target must expose logits for every input position. Export it with
all-position logits — torch2nnef `--num-logits-to-keep 0` (matching
transformers' `logits_to_keep == 0`) — so there is no last-token slice in the
graph, then load it with `from_paths_speculative`. Two drafters are provided:
`NgramDrafter` (prompt-lookup, no second model) and `ModelDrafter` (a smaller
causal LM).

```rust
let llm = CausalLlmModel::from_paths_speculative("tokenizer.json", "model.nnef.tgz", Default::default())?;
let mut state = llm.spawn()?;
state.append_text("The capital of France is")?;
let mut drafter = NgramDrafter::default();
while state.seq.len() < target_len {
    let stats = state.generate_speculative(&mut drafter, 4)?; // k = 4
}
```

Benchmarks:

- `cargo run --release --bin bench_spec -- -t tok.json -m model.nnef.tgz` —
  decode throughput, acceptance, and speedup of greedy vs n-gram speculation
  across prompt types and draft lengths (checks losslessness each run).
- `cargo run --release --bin bench_micro -- -t tok.json -m model.nnef.tgz` —
  forward latency vs tokens-per-pass, showing how much speculation a model can
  absorb before per-pass cost grows.

Notes: speculation helps most at high acceptance (repetitive / structured text)
and small `k`; a large vocabulary makes the all-position projection grow with
`k`, capping the useful draft length. Greedy speculation is lossless in exact
arithmetic; rare divergences on high-entropy text come from floating-point
differences between batched verification and single-token decode.
