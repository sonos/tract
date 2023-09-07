A simple example of training a Tensorflow model with Python, check the model with tract python API, then [loading it into `tract`](src/main.rs) and compare predictions.

# Python side training

```
pip install -r requirements.txt
```

Train a model, export it to ONNX along with a input and output example.

```
python example.py
```

(Outputs are commited to git, you don't need to run the python step at all.)

# Rust side inference

```
cargo run
```
