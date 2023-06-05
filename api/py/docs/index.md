# `tract` python bindings

`tract` is a library for neural network inference. While PyTorch and TensorFlow
deal with the much harder training problem, `tract` focuses on what happens once
the model in trained.

`tract` ultimate goal is to use the model on end-user data (aka "running the
model") as efficiently as possible, in a variety of possible deployments,
including some which are no completely mainstream : a lot of energy have been
invested in making `tract` an efficient engine to run models on ARM single board
computers.

## Getting started

### Install tract library

`pip install tract`. Prebuilt wheels are provided for x86-64 Linux and
Windows, x86-64 and arm64 for MacOS.

### Downloading the model

First we need to obtain the model. We will download an ONNX-converted MobileNET
2.7 from the ONNX model zoo.

`wget https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx`.

### Preprocessing an image

Then we need a sample image. You can use pretty much anything. If you lack
inspiration, you can this picture of Grace Hopper.

`wget https://s3.amazonaws.com/tract-ci-builds/tests/grace_hopper.jpg`

We will be needing `pillow` to load the image and crop it.

`pip install pillow`

Now let's start our python script. We will want to use tract, obviously, but we
will also need PIL's Image and numpy to put the data in the form MobileNet expects it.

```python
#!/usr/bin/env python

import tract
import numpy
from PIL import Image
```

We want to load the image, crop it into its central square, then scale this
square to be 224x224.

```python
im = Image.open("grace_hopper.jpg")
if im.height > im.width:
    top_crop = int((im.height - im.width) / 2)
    im = im.crop((0, top_crop, im.width, top_crop + im.width))
else:
    left_crop = int((im.width - im.height) / 2)
    im = im.crop((left_crop, 0, left_crop + im_height, im.height))
im = im.resize((224, 224))
im = numpy.array(im)
```

At this stage, we obtain a 224x224x3 tensor of 8-bit positive integers. We need to transform
these integers to floats and normalize them for MobileNet.
At some point during this normalization, numpy decides to promote our tensor to
double precision, but our model is single precison, so we are converting it
again after the normalization.

```python
im = (im.astype(float) / 255. - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
im = im.astype(numpy.single)
```

Finally, ONNX variant of Mobilenet expects its input in NCHW convention, and
our data is in HWC. We need to move the C axis before H and W, then insert the
N at the left.

```python
im = numpy.moveaxis(im, 2, 0)
im = numpy.expand_dims(im, 0)
```

### Loading the model

Loading a model is relatively simple. We need to instantiate the ONNX loader
first, the we use it to load the model. Then we ask tract to optimize the model
and get it ready to run.

```python
model = tract.onnx().model_for_path("./mobilenetv2-7.onnx").into_optimized().into_runnable()
```

If we wanted to process several images, this would only have to be done once
out of our image loop.

### Running the model

tract run methods take a list of inputs and returns a list of outputs. Each input
can be a numpy array. The outputs are tract's own Value data type, which should 
be converted to numpy array.

```python
outputs = model.run([im])
output = outputs[0].to_numpy()
```

### Interpreting the result

If we print the output, what we get is a array of 1000 values. Each value is
the score of our image on one of the 1000 categoris of ImageNet. What we want
is to find the category with the highest score.

```python
print(numpy.argmax(output))
```

If all goes according to plan, this should output the number 652. There is a copy
of ImageNet categories at the following URL, with helpful line numbering.

```
https://github.com/sonos/tract/blob/main/examples/nnef-mobilenet-v2/imagenet_slim_labels.txt
```

And... 652 is "microphone". Which is wrong. The trick is, the lines are
numbered from 1, while our results starts at 0, plus the label list includes a
"dummy" label first that should be ignored. So the right value is at the line
654: "military uniform". If you looked at the picture before you noticed that
Grace Hopper is in uniform on the picture, so it does make sense.

## Model cooking with `tract`

Over the years of `tract` development, it became clear that beside "training"
and "running", there was a third time in the life-cycle of a model. One of
our contributors nicknamed it "model cooking" and the term stuck. This extra stage
is about all what happens after the training and before running.

If training and Runtime are relatively easy to define, the model cooking gets a
bit less obvious. It comes from the realisation that the training form (an ONNX
or TensorFlow file or ste of files) of a model may is usually not the most
convenient form for running it. Every time a device loads a model in ONNX form
and transform it into a suitable form for runtime, it goes through the same
series or more or less complicated operations, that can amount to several
seconds of high-CPU usage for current models. When running the model on a
device, this can have several negative impact on experience: the device will
take time to start-up, consume a lot of battery energy to get ready, maybe fight
over CPU availability with other processes trying to get ready at the same
instant on the device.

As this sequence of operations is generally the same, it becomes relevant to
persist the model resulting of the transformation. It could be persisted at the
first application start-up for instance. But it could also be "prepared", or
"cooked" before distribution to the devices.

## Cooking to NNEF

`tract` supports NNEF. It can read a NNEF neural network and run it. But it can
also dump its preferred representation of a model in NNEF.

At this stage, a possible path to production for a neural model becomes can be drawn:
* model is trained, typically on big servers on the cloud, and exported to ONNX.
* model is cooked, simplified, using `tract` command line or python bindings.
* model is shipped to devices or servers in charge of running it.

## Testing and benching models early

As soon as the model is in ONNX form, `tract` can load and run it. It gives
opportunities to validate and test on the training system, asserting early on that
`tract` will compute at runtime the same result than what the training model
predicts, limiting the risk of late-minute surprise.

But tract command line can also be used to bench and profile an ONNX model on
the target system answering very early the "will the device be fast enough"
question. The nature of neural network is such that in many cases an
untrained model, or a poorly trained one will perform the same computations than
the final model, so it may be possible to bench the model for on-device
efficiency before going through a costly and long model training.

## tract-opl

NNEF is a pretty little standard. But we needed to go beyond it and we extended
it in several ways. For instance, NNEF does not provide syntax for recurring
neural network (LSTM and friends), which are an absolute must in signal and voice
processing. `tract` also supports symbolic dimensions, which are useful to
represent a late bound batch dimension (if you don't know in advance how many
inputs will have to be computed concurrently).

## Pulsing

For interactive applications where time plays a role (voice, signal, ...),
`tract` can automatically transform batch models, to equivalent streaming models
suitable for runtime. While batch models are presented at training time the
whole signal in one go, a streaming model received the signal by "pulse" and
produces step by step the same output that the batching model.

It does not work for every model, `tract` can obviously not generate a model
where the output at a time depends on input not received yet. Of course, models
have to be *causal* to be pulsable. For instance, a bi-directional LSTM is not
pulsable. Most convolution nets can be made causal at designe time by padding,
or at cooking time by adding fixed delays.

This cooking step is a recurring annoyance in the real-time voice and signal
field : it can be done manually, but is very easy to get wrong. `tract` makes
it automactic.
