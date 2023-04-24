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

### Install tract

Just `pip install tract`. Prebuilt wheels are provided for x86-64 Linux and
Windows, x86-64 and arm64 for MacOS.



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

NNEF is another standard for neural networks exchange. While ONNX is designed
for models at training time, NNEF focused on the simpler semantics at play at
runtime. Unencumbered by their training semantics, many layers collapse into a
smaller operators set, made of arithmetic operations and simple tensor
transformations.

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
