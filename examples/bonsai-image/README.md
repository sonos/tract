# Bonsai Image with tract

Text to image with PrismML's Ternary Bonsai Image 4B, a Qwen3 text encoder, a ternary Flux.2
Klein transformer and its VAE, on tract's `wgpu` runtime, in the browser and natively.

The weights stay packed on the GPU: `Q2_0_T` for the transformer's ternary matmuls, `Q4_0` for
the encoder's. The NNEF archives are the exports in
[tract-nnef/bonsai-image-ternary-4B](https://huggingface.co/tract-nnef/bonsai-image-ternary-4B),
which take the image size as a symbolic dimension, so one archive serves every size. The encoder
is its GPTQ-calibrated one.

The browser page is in [web](web), and live at <https://demo.tract.rs/bonsai-image/>. The same
pipeline runs natively, to check its images:

```sh
huggingface-cli download tract-nnef/bonsai-image-ternary-4B transformer.nnef.tar text_encoder_gptq.nnef.tar vae_decoder.nnef.tar --local-dir models
huggingface-cli download prism-ml/bonsai-image-ternary-4B-unpacked tokenizer/tokenizer.json --local-dir models
mv models/tokenizer/tokenizer.json models/
cargo run --release -p bonsai-web --bin bonsai-native -- models "a photo of a bonsai tree" bonsai.png
```

`BONSAI_SIZE` (`HxW`, multiples of 16, `256x256` by default), `BONSAI_STEPS` (4) and
`BONSAI_SEED` (42) set the rest. A seed gives the same image natively and in the browser.
