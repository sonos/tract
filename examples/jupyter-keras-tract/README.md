A simple example of training a Tensorflow model with Python in a Jupyter notebook, then loading it into `tract` to make predictions.

With shape optimizations:
```
time cargo run
real    0m1.145s
user    0m0.172s
sys     0m0.219s
```

Without optimizations:
```
time targo run
real    0m0.280s
user    0m0.047s
sys     0m0.219s
```

Python
```
time python make_predictions.py
real    0m2.388s
user    0m2.266s
sys     0m1.859s
```