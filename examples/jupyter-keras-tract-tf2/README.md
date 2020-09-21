A simple example of training a Tensorflow model with Python in a [Jupyter notebook](simple_model.ipynb), then [loading it into `tract`](src/main.rs) to make predictions.

# To Use
`conda env create -f environment.yml`

Run the Jupyter notebook, which will create the model artifacts (1 onnx for `tract`, and one tensorflow artifact for benchmarking)

Python

```
time python make_predictions.py
real    0m1.667s
user    0m2.575s
sys     0m1.301s
```

tract, even in debug mode, is significantly faster:

```
time cargo run
real    0m0.111s
user    0m0.080s
sys     0m0.040s
```

In a real-life server settings, the model would be loaded and optimized only once and used repeastedly to make predictions on different inputs. Compiled in release mode, the call to `run()` is clocked at 6 microseconds (0m0.000006s !) on one single core.
