Example of exporting a PyTorch to ONNX format, then performing inference with
tract.

**Export Model:**
```bash
python export.py
```

**Inference on `elephants.jpg`:**
```
cargo run
result: Some((22.08386, 102))
```

Predicts class 102 (`tusker`).
