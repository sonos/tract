# FLUX.1-schnell — WIP status

## What works

End-to-end image generation from text prompt, running on tract with CUDA backend.

```
flux-schnell -p "a photo of a cat" -s 4 --seed 42
```

- **Text encoders** (CLIP-L + T5-XXL): f16 ONNX, validated against PyTorch reference (`--approx ultra`)
- **Transformer** (12B params): f16 ONNX, validated (`--approx-custom 0.5,0.5,0.001` — 0.06% outliers due to GPU f16 rounding differences)
- **VAE decoder**: **f32** ONNX, validated (`--approx ultra`)
- **Rust pipeline** (`src/main.rs`): sequential load/run/unload per component to stay within 32GB VRAM

## Memory management

The full pipeline (T5-XXL ~22GB + transformer ~24GB in f16, plus VAE ~300MB in f32) cannot
fit in 32GB VRAM simultaneously. Both the Python export scripts and the Rust pipeline handle
this by processing one component at a time.

### Python export (`export.py`)
The original script loaded `FluxPipeline` which pulls all components into RAM at once (~93GB
in f32). This crashed the box twice (OOM, kernel freeze, power cycle). Rewritten to:
- Load each sub-model individually via `from_pretrained(..., subfolder=...)`
- Export to ONNX, then `del` + `gc.collect()` before loading the next
- Peak RAM ~36GB (the transformer), well within 62GB

### Rust pipeline (`src/main.rs`)
Each model is loaded, run, and dropped in its own scope block:

1. **CLIP-L text encoder** — loaded on **CPU**, run, dropped. Output: `pooled` (1,768) f16
2. **T5-XXL text encoder** — loaded on **CPU**, run, dropped. Output: `t5_embeds` (1,256,4096) f16
3. **Transformer** (12B) — loaded on **GPU** (CUDA), runs the 4-step denoising loop, dropped.
   Inputs/outputs are f16, but the latent accumulation (`latent += pred * dt`) is done in f32
   on the CPU side to avoid precision loss over multiple steps.
4. **VAE decoder** — loaded on **GPU** (CUDA), decodes each image, dropped.
   Runs in f32 (instance norm overflows in f16).

Text encoders run on CPU because they're fast enough and this avoids competing for VRAM with
the transformer. The transformer is the bottleneck (~2.5s/step on RTX 5090).

## What was built along the way

These changes live on the `cuda-conv-f16` base branch:

1. **CUDA f16 conv generic kernel** — templated `cnn.cu` over f32/f16, accumulates in f32
2. **cuDNN f16 conv** — for 2D convs (cuDNN rejects f16 3D convs); 3D+ falls back to generic
3. **conv_f16 test suite** — `ConvProblemF16` wrapper in `test-rt/suite-unit`, registered for CUDA
4. **Removed group restriction** from CUDA conv tests (was unnecessary)
5. **LayerNorm f16 fix** — cast scale/bias back to input dtype before final mul/add
6. **Gemm f16 fix** — cast alpha/beta constants to input dtype

## VAE f16 — known issue

VAE decoder produces all-NaN in f16. Likely cause: instance normalization sums over 1024x1024 spatial dims, overflowing f16 range (~65504 max). Tracked in `/workspace/todolist.md`. Possible fix: compute norm statistics in f32 (stash_type approach) while keeping data in f16.

## How to reproduce

### Prerequisites
- HuggingFace account with access to `black-forest-labs/FLUX.1-schnell`
- GPU with >=32GB VRAM (tested on RTX 5090)

### Export models
```bash
python3 -m venv .venv
.venv/bin/pip install torch diffusers transformers onnx onnxscript sentencepiece protobuf
.venv/bin/huggingface-cli login
.venv/bin/python export.py        # exports 4 ONNX models to assets/
.venv/bin/python reference.py     # generates .io.npz reference bundles
```

### Validate with tract-cli
```bash
TRACT=../../target/opt-no-lto/tract
$TRACT assets/text_encoder.onnx --cuda run --allow-float-casts --input-from-bundle assets/text_encoder.io.npz --assert-output-bundle assets/text_encoder.io.npz --approx ultra
$TRACT assets/text_encoder_2.onnx --cuda run --allow-float-casts --input-from-bundle assets/text_encoder_2.io.npz --assert-output-bundle assets/text_encoder_2.io.npz --approx ultra
$TRACT assets/transformer.onnx --cuda run --allow-float-casts --input-from-bundle assets/transformer.io.npz --assert-output-bundle assets/transformer.io.npz --approx-custom 0.5,0.5,0.001
$TRACT assets/vae_decoder.onnx --cuda run --input-from-bundle assets/vae_decoder.io.npz --assert-output-bundle assets/vae_decoder.io.npz --approx ultra
```

### Run Rust example
```bash
cargo run --profile opt-no-lto -p flux-schnell -- --assets assets -p "a photo of a cat"
```
