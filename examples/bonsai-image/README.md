# Bonsai Image with tract

Text to image with PrismML's Ternary Bonsai Image 4B, a Qwen3 text encoder, a ternary Flux.2
Klein transformer and its VAE, on tract's `wgpu` runtime.

The weights stay packed on the GPU: `Q2_0_T` for the transformer's ternary matmuls, `Q4_0` for
the encoder's. The transformer's weights are ternary as published; the encoder's are quantized
to 4 bits by `export.py`.

## Native

```sh
pip install torch diffusers transformers accelerate safetensors onnx onnxscript pillow
huggingface-cli download prism-ml/bonsai-image-ternary-4B-unpacked --local-dir hf
ASSETS=assets HF_SNAPSHOT=hf python export.py
cp -r hf/tokenizer assets/
cargo run --release -p bonsai-image -- -p "a photo of a bonsai tree"
```

`SIZE=HxW` picks the size, 256x256 by default, with both multiples of 16. `DIT_TEXT_LEN` trims
the text tokens the transformer takes, which the demo must then be told with `--text-len`.

`--staged` runs a load-then-free stand-in of the three stages on tiny graphs.

## In the browser

See [web](web).
