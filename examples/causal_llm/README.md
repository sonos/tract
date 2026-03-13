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
