# Tract Example: Translating to NNEF 

Using NNEF instead of TensorFlow or ONNX allow for faster loading time and
smaller binaries, but require precompilation of the networks.

This example shows how to translate a Neural Network to NNEF (with extensions
if needed), then use this translated network for prediction.

```sh
git clone https://github.com/snipsco/tract
cd tract/examples/nnef-mobilenet-v2/
```

## Installing tract command line tool

This one is going to take a while.

```sh
cargo install tract
```

## Obtaining the model 

MobileNet is a response to the ImageNet challenge. The goal is to categorize
images and associate them with one of 1000 labels. In other words, recognize a
dog, a cat, a rabbit, or a military uniform.

You will need to download the models. For instance:

```sh
wget https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz
tar zxf mobilenet_v2_1.4_224.tgz
```

This expands a half-dozen files in the directory. The only one of interest
for us is the frozen TensorFlow model: `mobilenet_v2_1.4_224_frozen.pb`.

## Converting the network

We need to tell tract about the input shape and types. TensorFlow uses the NHWC
convention, and the variant of Mobilenet we picked operates on inputs of
224x224 pixels:

```sh
tract mobilenet_v2_1.4_224_frozen.pb -i 1,224,224,3,f32 dump --nnef mobilenet.nnef.tgz
```

## Running with tract_nnef

```
use tract_nnef::prelude::*;

fn main() -> TractResult<()> {
    let model = tract_nnef::nnef()
        .model_for_path("mobilenet.nnef.tgz")?
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
    Ok(())
}
```

Compared to running the original mobilenet TensorFlow model (see
[../tensorflow-mobilenet-v2](tensorflow example), there is no longer need to
give the input size hint (they are now embedded in the graph description).
Similarly, the InferenceModel and InferenceFact have disappeared: we no longer
need to operate with partially typed networks.
