# Tract examples: Tensorflow MobileNet v2

This project is a simple project with minimal code showing how to use tract to
process an image with MobileNetV2.

The example assume the following command are run in the directory of this
example project, where this README lives.

```
git clone https://github.com/snipsco/tract
cd  tract/examples/tensorflow-mobilenet-v2/
```

## Obtaining the model 

See https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet for more information.

These files are too big to be included on the github repository, so you will need to download them.

For instance:

```
wget https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz
tar zxf mobilenet_v2_1.4_224.tgz
```

This will expand a half-dozen files in the directory. The only one of interest for us is the frozen TensorFlow model: `mobilenet_v2_1.4_224_frozen.pb`.

## Converting the input image

As in TensorFlow documentation, we will use a portrait of Grace Hopper.

```
grace_hopper.jpg: JPEG image data, JFIF standard 1.02, resolution (DPI), density 96x96, segment length 16, baseline, precision 8, 517x606, components 3
```

## Try it

`cargo run` should print a lot of things, and ultimately: `result: Some((653, 0.32560226))`.

This is actually good. It is the rank and a confidence indicator of the
inferred label.

```
$ cat -n imagenet_slim_labels.txt | grep -C 3 653
   650  medicine chest
   651  megalith
   652  microphone
   653  microwave
   654  military uniform
   655  milk can
   656  minibus
```

There is a one-off error as the 1000 "real" labels list includes a "dummy"
label at its top, but tract and mobilenet correctly found out about Grace
Hopper military uniform.

## A look at the code

Everything happens in [src/main.rs](src/main.rs). It uses three crates:

* `tract-core` is the main tract crate. It contains tract operators, model
    analysing and running infrastructure.
* `tract-tensorflow` is the TensorFlow format parser. It translates most of
    TensorFlow operators to core ones, or implements the one which are specific
    to TensorFlow.
* finally we use the `image` crate to load and resize the JPEG portrait.


### Loading the model

This line creates a tract-tensorflow context, and uses it to load the protobuf model.

```rust
    let mut model =
        tract_tensorflow::tensorflow().model_for_path("mobilenet_v2_1.4_224_frozen.pb")?;
```

### Specifying input size and optimizing.

TensorFlow models typically do not specify explicitely the input dimensions,
but a lot of optimization in `tract` depends on the knownledge of all tensors
types and shapes in the network.

MobileNet assumes its input in in the NHWC convention: [batch, height, width,
channels]. The MobileNet variant we have picked works with a 224x224 square
pictures of RGB pictures (C=3), we will only process one image at a time (N=1).
It operates on single precision floats.

```rust
    model.set_input_fact(0, TensorFact::dt_shape(f32::datum_type(), tvec!(1, 224, 224, 3)))?;

    let model = model.into_optimized()?;
    let plan = SimplePlan::new(&model)?;
```

Now the model is ready to run, we have an execution plan, so let's prepare the
image.

### Conditioning the input

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
    let result = plan.run(tvec!(image))?;
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
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    println!("result: {:?}", best);
```
