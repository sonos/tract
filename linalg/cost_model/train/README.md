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
best one to `neural_net_a53.npz`, run:

```sh
python train.py -N 15 --platform=a53 a53-dataset neural_net_a53.npz
```

This will save the neural network in an archive with keys:

  - `input.mean` and `input.std`: normalization parameter for the input features
  - `linear_1.weight` and `linear_1.bias`: the weight and biases for the first layer
  - `linear_2.weight` and `linear_2.bias`: the weight and biases for the first layer
  - `kernels`: the output classes

The neural network computes

```
  softmax( linear_2.bias + linear_2.weight * tanh(linear_1.bias + linear_1.weight * (x - input.mean) / input.std ) )
```
