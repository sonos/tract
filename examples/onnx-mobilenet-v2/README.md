# Tract examples: ONNX MobileNet v2

This project is a simple project with minimal code showing how to use tract to
process an image with MobileNetV2.

The example assume the following command are run in the directory of this
example project, where this README lives.

```sh
git clone https://github.com/snipsco/tract
cd tract/examples/onnx-mobilenet-v2/
```

## Obtaining the model 

MobileNet is a response to the ImageNet challenge. The goal is to categorize
images and associate them with one of 1000 labels. In other words, recognize a
dog, a cat, a rabbit, or a military uniform.

See https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet for more information.

Models can get a big heavy, so we chose not to include them in tract git repository. 
You will need to download the models. For instance:

```sh
wget https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx
```

## Input image

We will use a portrait of Grace Hopper in uniform (included in the repository).

```
grace_hopper.jpg: JPEG image data, JFIF standard 1.02, resolution (DPI), density 96x96, segment length 16, baseline, precision 8, 517x606, components 3
```

## Try it

`cargo run` should print a lot of things, and ultimately: `result: Some((11.4773035, 654))`.

This is actually good. It is the rank (654) and a confidence indicator (11.4773035)
of the inferred label.

```
$ cat -n imagenet_slim_labels.txt | grep -C 3 654
   651  megalith
   652  microphone
   653  microwave
   654  military uniform
   655  milk can
   656  minibus
   657  miniskirt
```

## A look at the code

Everything happens in [src/main.rs](src/main.rs).


```rust
use tract_onnx::prelude::*;

fn main() -> TractResult<()> {
    let model = tract_onnx::onnx()
        // load the model
        .model_for_path("mobilenetv2-7.onnx")?
        // specify input type and shape
        .with_input_fact(0, f32::fact(1, 3, 224, 224).into())?
        // optimize the model
        .into_optimized()?
        // make the model runnable and fix its inputs and outputs
        .into_runnable()?;

    // open image, resize it and make a Tensor out of it
    let image = image::open("grace_hopper.jpg").unwrap().to_rgb8();
    let resized =
        image::imageops::resize(&image, 224, 224, ::image::imageops::FilterType::Triangle);
    let image: Tensor = tract_ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
        let mean = [0.485, 0.456, 0.406][c];
        let std = [0.229, 0.224, 0.225][c];
        (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
    })
    .into();

    // run the model on the input
    let result = model.run(tvec!(image))?;

    // find and display the max value with its index
    let best = result[0]
        .to_array_view::<f32>()?
        .iter()
        .cloned()
        .zip(2..)
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    println!("result: {:?}", best);
    Ok(())
}
```

It uses the tract-onnx as an entry point. Other options are available (tensorflow, or nnef):
I also use the `image` crate to load and resize the JPEG portrait.

### Loading the model

This line creates a tract-onnx context, and uses it to load the protobuf
model.

```rust
    let model = tract_onnx::onnx()
        .model_for_path("mobilenet_v2_1.4_224_frozen.pb")?
    // ..
```

### Specifying input size and optimizing.

Modelsa often do not specify explicitely the input dimensions,
but a lot of optimization in `tract` depends on the knownledge of all tensors
types and shapes in the network.

Onnx variant of MobileNet assumes its input in in the NCHW convention: 
[batch, channels, height, width]. The MobileNet variant we have picked works with a 224x224 square
RGB (C=3) pictures. We will only process one image at a time (N=1).
And it operates on single precision floats (aka `f32`).

```rust
    // ..
        .with_input_fact(0, f32::fact(&[1, 3, 224, 224]).into())?
        .into_optimized()?
        .into_runnable()?;
```

Now the model is ready to run, we have an execution plan, so let's prepare the
image.

### Conditioning the input

We use the `image` crate to load the `.jpg` image, resize is to 224x224. Then
we build an 4-dimension array in the right NHWC shape, with `f32` obtained by
normalizing the `u8` input to the `0..1` range. This array is then converted
into a Tensor. We apply a color normalization on the fly, which is standard for
MobileNet models.

```rust
    let image = image::open("grace_hopper.jpg").unwrap().to_rgb();
    let resized = image::imageops::resize(&image, 224, 224, ::image::FilterType::Triangle);
    let image: Tensor = tract_ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
        let mean = [0.485, 0.456, 0.406][c];
        let std = [0.229, 0.224, 0.225][c];
        (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
```

Note that `tract` crates re-export the excellent `ndarray` crate as `tract_ndarray`so that 
it is easy to get the right version for tract conversions to work.

### Run the network!

```rust
    let result = model.run(tvec!(image))?;
```

### Interpret the result

Finally we grab the single Tensor output by the plan execution, convert it to a
ndarray ArrayView of f32 values. It is a single dimension (a vector...) of 1001
category scores (1000 labels plus the dummy one). We need pick the maximum
score, with its index, and diplay it...

```rust
    let best = result[0]
        .to_array_view::<f32>()?
        .iter()
        .cloned()
        .enumerate()
        .zip(1..)
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    println!("result: {:?}", best);
```
