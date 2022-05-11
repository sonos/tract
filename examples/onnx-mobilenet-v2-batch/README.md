# Tract examples: ONNX MobileNet v2 optimized for batch

This project demounstrate how to use symbolic dimensions in tract while 
performing network optimisation.
It builds upon the onnx-mobilenet-v2 example, but here we will optimise 
the network without assuming it will always run on one single image.

## Assets

Model is the same as in the quick start example:

```sh
wget https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx
```

We will use Grace Hopper portrait again, and a cat image.

## Loading and optimising the model

```rust
    let batch = Symbol::new('N');
    let model = tract_onnx::onnx()
        .model_for_path("mobilenetv2-7.onnx")?
        .with_input_fact(0, f32::fact(dims!(batch, 3, 224, 224)).into())?
        .with_output_fact(0, InferenceFact::default())?
        .into_optimized()?
        .into_runnable()?;
```

The differences in model preparation is the most tricky part: we need to introduce
a "variable" `N` to be use in shape computation. It represent the batch size, it's the
N from NCHW. We could actually use any letter or charater, tract does not really know
anything specific about batch size.

`f32::fact(dims!(batch, 3, 224, 224)).into()`.

The shape definition, while a bit complex syntax-wise, just says "N,3,224,224".

One other trick is... this network actually encodes a batch size in two places in the
protobuf file: on the input (which we just override with our N-based input fact) but also
in its output. So we need to "erase" that output shape information:

`.with_output_fact(0, InferenceFact::default())?`

We could also define it as `N,1001`, but we just let tract shape inference do the work.

## Loading images and running

The rest is pretty straightforward. We make a tensor with two images, but we could also
use the same optimised network with one image or any number...
