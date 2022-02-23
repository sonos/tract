# Matrix-matrix multiplication kernel prediction

Script to train a feed-forward neural network with one hidden layer to
predict which kernel to use for matrix-matrix multiplication, from a dataset
of measurements made with tract.

To install the dependencies in a virtual environment:

```sh
virtualenv venv
pip install -r requirements.txt
```

To train `N=15` neural networks on the dataset (e.g. `a53-dataset`) and save the
best one to `neural_net_a53.rs`, run:

```sh
python train.py -N 15 --platform=a53 a53-dataset neural_net_a53.rs
```

This will save the neural network as an instance of ../../src/frame/mmm/cost_model.rs.

The neural network computes:

```
  softmax( b2 + w2 * tanh(b1 + w1 * (x - feat_norm_mean) / feat_norm_stddev ) )
```

Rust CostModel implementation is for kernel selection. It is only interested in the ArgMax, so it skips the SoftMax.
