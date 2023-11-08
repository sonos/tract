# Tract examples: TensorflowLite MobileNet v3

This project is a simple project with minimal code showing how to use tract to
process an image with MobileNetV3.

The example assume the following command are run in the directory of this
example project, where this README lives.

```sh
git clone https://github.com/snipsco/tract
cd tract/examples/tflite-mobilenet-v3
```

## Obtaining the model 

MobileNet is a response to the ImageNet challenge. The goal is to categorize
images and associate them with one of 1000 labels. In other words, recognize a
dog, a cat, a rabbit, or a military uniform.

See https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet for more information.

You will need to download the models. For instance:

```sh
wget -q https://tfhub.dev/google/lite-model/imagenet/mobilenet_v3_small_100_224/classification/5/default/1?lite-format=tflite -O mobilenet_v3_small_100_224.tflite
```

## Converting the input image

As in TensorFlow documentation, we will use a portrait of Grace Hopper
(included with this example).

```
grace_hopper.jpg: JPEG image data, JFIF standard 1.02, resolution (DPI), density 96x96, segment length 16, baseline, precision 8, 517x606, components 3
```

## Try it

`cargo run` should print a lot of things, and ultimately: `result: Some((0.32560226, 654))`.

This is actually good. It is the rank (654) and a confidence indicator (0.32)
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
    let model = tract_tflite::tflite()
        // load the model
        .model_for_path("./mobilenet_v3_small_100_224.tflite")?
        // optimize the model
        .into_optimized()?
        // make the model runnable and fix its inputs and outputs
        .into_runnable()?;

    // open image, resize it and make a Tensor out of it
    let image = image::open("grace_hopper.jpg").unwrap().to_rgb();
    let resized =
        image::imageops::resize(&image, 224, 224, ::image::imageops::FilterType::Triangle);
    let image: Tensor = tract_ndarray::Array4::from_shape_fn((1, 224, 224, 3), |(_, y, x, c)| {
        resized[(x as _, y as _)][c] as f32 / 255.0
    })
    .into();

    // run the model on the input
    let result = model.run(tvec!(image.into()))?;

    // find and display the max value with its index
    let best = result[0]
        .to_array_view::<f32>()?
        .iter()
        .cloned()
        .zip(1..)
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    println!("result: {:?}", best);
```

It uses three crates:

* `tract-core` is the main tract crate. It contains tract operators, model
    analysing and running infrastructure.
* `tract-tflite` is the TensorFlow format parser. It translates most of
    TensorFlow operators to core ones, or implements the one which are specific
    to TensorFlow.
* finally we use the `image` crate to load and resize the JPEG portrait.


### Loading the model

This line creates a tract-tensorflow context, and uses it to load the protobuf
model.

```rust
    let model = tract_tflite::tflite()
        .model_for_path("./mobilenet_v3_small_100_224.tflite")?
        .into_optimized()?
        .into_runnable()?;
```

Now the model is ready to run, we have an execution plan, so let's prepare the
image.

### Conditioning the input

MobileNet assumes its input in in the NHWC convention: [batch, height, width,
channels]. The MobileNet variant we have picked works with a 224x224 square
RGB (C=3) pictures. We will only process one image at a time (N=1).
And it operates on single precision floats (aka `f32`).

We use the `image` crate to load the `.jpg` image, resize is to 224x224. Then
we build an 4-dimension array in the right NHWC shape, with `f32` obtained by
normalizing the `u8` input to the `0..1` range. This array is then converted
into a Tensor.

```rust
    let image = image::open("grace_hopper.jpg").unwrap().to_rgb();
    let resized = image::imageops::resize(&image, 224, 224, ::image::FilterType::Triangle);
    let image: Tensor = ndarray::Array4::from_shape_fn((1, 224, 224, 3), |(_, y, x, c)| {
        resized[(x as _, y as _)][c] as f32 / 255.0
    }).into();
```

Note that `tract-core` re-export the excellent `ndarray` crate so that it is
easy to get the right version for tract conversion to work.

### Run the network!

```rust
    let result = model.run(tvec!(image.into()))?;
```

### Interpret the result

Finally we grab the single Tensor output by the plan execution, convert it to a
ndarray ArrayView of f32 values. It is a single dimension (a vector...) of 1001
category scores (1000 labels plus the dummy one). We need pick the maximum
score, with its index, and display it...

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
