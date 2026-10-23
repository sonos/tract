# Stable Diffusion 1.5 with tract

A minimal text-to-image pipeline running entirely in Rust using [tract](https://github.com/sonos/tract). All three SD 1.5 components (text encoder, UNet, VAE decoder) run on GPU via tract's CUDA or Metal backend, falling back to CPU.

## Quick start

```bash
# Export ONNX models and tokenizer (one-time setup)
pip install torch diffusers transformers accelerate onnxscript onnx Pillow
python export.py

# Generate images
cargo run --release -- -p "a photo of a cat" -o cat.png
```

## Usage

```
stable-diffusion [OPTIONS]

Options:
  -p, --prompt <PROMPT>            Text prompt [default: "a photo of a cat"]
  -n, --num-images <NUM_IMAGES>    Number of images [default: 1]
  -s, --steps <STEPS>              Denoising steps [default: 20]
      --seed <SEED>                Random seed [default: 42]
  -g, --guidance-scale <SCALE>     CFG scale [default: 7.5]
  -o, --output <OUTPUT>            Output file [default: output.png]
      --assets <ASSETS>            Model directory [default: assets]
      --f32                        Run the UNet in f32 (f16 on GPU otherwise)
  -h, --help                       Print help
```

Multiple images get numbered: `output_0.png`, `output_1.png`, etc.

Images display inline on iTerm2/WezTerm (including through tmux).

## Performance

On an RTX 5090, wall clock including model loading (~4 s):

| Images | Steps | Time | Denoising |
|--------|-------|------|-----------|
| 1 | 20 | ~4.4s | ~28 ms/step |
| 4 | 20 | ~5.4s | ~69 ms/step |
| 1 | 10 | ~4.2s | ~35 ms/step |

All models go through `transformers_detect_all` (fused attention). The UNet runs in f16 on GPU runtimes, the text encoder and VAE in f32. The UNet uses batched inference for classifier-free guidance (batch=2N for N images).

## Architecture

```
Prompt → [Tokenizer] → [CLIP Text Encoder] → text embeddings
                                                    ↓
Random noise → [Euler Scheduler + UNet × steps] → denoised latent
                                                    ↓
                              [VAE Decoder] → 512×512 RGB image
```

- **Tokenizer**: HuggingFace `tokenizers` crate (loads `tokenizer.json`)
- **Text Encoder**: CLIP ViT-L/14, ONNX, runs on GPU
- **UNet**: SD 1.5, ONNX with dynamic batch, runs on GPU
- **VAE Decoder**: SD 1.5, ONNX, runs on GPU
- **Scheduler**: Euler discrete, computed in Rust from the SD 1.5 noise schedule constants

## Model export

`export.py` exports the three model components from HuggingFace diffusers to ONNX. The UNet is exported with dynamic batch axes so tract can run classifier-free guidance in a single batched call.

`reference.py` generates per-model input/output reference data for validation. The CI script validates each model against these references using `tract --assert-output-bundle`.

## Files

| File | Description |
|------|-------------|
| `src/main.rs` | Rust pipeline: tokenize, encode, denoise, decode, save |
| `export.py` | Export SD 1.5 to ONNX |
| `reference.py` | Generate validation data |
| `ci.sh` | End-to-end CI: export + validate + generate |
| `assets/` | Models, tokenizer (generated, not checked in) |
