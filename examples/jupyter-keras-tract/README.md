A simple example of training a Tensorflow model with Python in a [Jupyter notebook](simple_model.ipynb), then [loading it into `tract`](src/main.rs) to make predictions. Specifically, the notebook shows how to convert a keras / TensorFlow 2 model to a TensorFlow 1 format: `tract` does not support TensorFlow 2 much more complex format at this point.

Python

```
time python make_predictions.py
real    0m2.388s
user    0m2.266s
sys     0m1.859s
```

tract, even in debug mode is faster:

```
time cargo run
real    0m0.280s
user    0m0.047s
sys     0m0.219s
```

In a real-life server settings, the model would be loaded and optimized only once and used repeastedly to make predictions on different inputs. Compiled in release mode, the call to `run()` is clocked at 6 microseconds (0m0.000006s !) on one single core.
