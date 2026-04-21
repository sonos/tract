# tract examples

Small end-to-end programs demonstrating how to load and run models with
tract. Most use the high-level [`tract`](../api/rs) facade; a few still
sit on the lower-level framework crates (`tract-tensorflow`,
`tract-tflite`) and are flagged below.

## Image classification

Simple vision pipelines: load a model, feed a pre-processed image tensor,
print the top class. Useful entry points when you're just kicking tract's
tires.

| Example | Model source | Notes |
|---------|--------------|-------|
| [`onnx-mobilenet-v2`](onnx-mobilenet-v2) | ONNX | MobileNet v2. Second binary `dyn-shape` demonstrates symbolic batch dim and runtime shape dispatch. |
| [`nnef-mobilenet-v2`](nnef-mobilenet-v2) | NNEF | Same model, pre-compiled to NNEF. Faster load, smaller artifact. |
| [`nnef-dump-mobilenet-v2`](nnef-dump-mobilenet-v2) | TF → NNEF | Shows how to `tract … dump --nnef` a TF model and then run the NNEF. |
| [`pytorch-resnet`](pytorch-resnet) | PyTorch → ONNX | ResNet classification with `ndarray` broadcasting for per-channel normalisation. |
| [`keras-tract-tf2`](keras-tract-tf2) | Keras → ONNX | Numeric equivalence check: runs the same inputs through the ONNX model and compares to `.npz` reference outputs. |
| [`tensorflow-mobilenet-v2`](tensorflow-mobilenet-v2) | TensorFlow `.pb` | Uses the legacy `tract-tensorflow` crate directly — the `tract` facade does not expose TF `.pb` loading. |
| [`tflite-mobilenet-v3`](tflite-mobilenet-v3) | TFLite | Uses the legacy `tract-tflite` crate directly — the `tract` facade does not expose TFLite loading. |

## Detection and face similarity

| Example | What it shows |
|---------|---------------|
| [`face_detection_yolov8onnx_example`](face_detection_yolov8onnx_example) | YOLOv8-face ONNX model, letterbox pre-processing, bounding-box decoding. |
| [`face_similarity_arcface_onnx`](face_similarity_arcface_onnx) | Two-stage pipeline: YOLOv8-face for detection, ArcFace for embedding, cosine similarity between two images. |

## NLP

| Example | What it shows |
|---------|---------------|
| [`pytorch-albert-v2`](pytorch-albert-v2) | ALBERT masked-LM token completion; demonstrates `tract`'s tokenizer integration and tuple-argument `run(...)`. |
| [`causal_llm`](causal_llm) | Reusable crate + CLI + OpenAI-compatible HTTP server for NNEF-packaged causal LLMs (Llama-style). Exercises KV-cache unfolding, prompt chunking, repeat-penalty sampling, state freeze/truncate. |

## Speech recognition

NVIDIA NeMo RNN-Transducer-style ASR pipelines: preprocessor + encoder +
decoder + joint network driven as four separate tract models.

| Example | What it shows |
|---------|---------------|
| [`nemo-parakeet-asr`](nemo-parakeet-asr) | Parakeet offline ASR on a WAV file; showcases `tract::nnef()` loading and CPU/GPU runtime selection. |
| [`nemo-nemotron-asr`](nemo-nemotron-asr) | Nemotron ASR with a `transformers_detect_all` transform pass and a `patch` transform to prune a variable-length input. |

## Image generation (diffusion)

End-to-end text-to-image pipelines with CLIP / T5 text encoders,
transformer / UNet backbones, and VAE decoders. All three pick the best
available runtime (`cuda` → `metal` → cpu).

| Example | What it shows |
|---------|---------------|
| [`stable-diffusion`](stable-diffusion) | SD 1.5 txt2img; Euler scheduler built from scratch, classifier-free guidance, 512×512 output. |
| [`stable-diffusion-xl`](stable-diffusion-xl) | SDXL 1.0 txt2img; two text encoders, pooled + hidden-state embeddings, time_ids, 1024×1024 output. |
| [`stable-diffusion-3`](stable-diffusion-3) | SD 3 Medium txt2img; MMDiT transformer backbone, optional T5-XXL text encoder, flow-matching scheduler. |

## Ndarray interop

Examples that consume or produce `ndarray::ArrayD` values call
`tract::impl_ndarray_interop!()` at their crate root. The macro expands in
the caller's scope, so the `ndarray` version used is whichever one the
example declares in its `Cargo.toml` — `tract` itself no longer re-exports
`ndarray`. See the macro's rustdoc or `onnx-mobilenet-v2/src/main.rs` for
the basic pattern.
