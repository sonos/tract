# tract cli recipes

`tract` command line is meant to be an auditing, debugging, profiling tool, for CI and
interactive usage.

Please do not make assumptions on the exact forms of its outputs. We do not commit on
any form of stability suitable to script writing.

We are going to use [ONNX mobilenet](../examples/onnx-mobilenet-v2) as examples in these notes. See its
code and README to download a the model.

## Install `tract`

* build latest release: 

```
cargo install tract
```

* download a prebuilt binary fo MacOs Intel, linux Intel, Armv7 or Armv8

```
https://github.com/sonos/tract/releases/latest
```

* run from source

```
git clone https://github.com/sonos/tract
cd cli
cargo run
```

## Model loading

First, `tract` needs to load a model and pipeline it to the preferred `tract-opl` form.
This is equivalent to using the API to load an onnx file, and setting input shapes with
InferenceFact and with_input_fact.

```bash
tract mobilenetv2-7.onnx -i 1,3,224,224
```

loads the model like:

```rust
tract_onnx::onnx()
            .model_for_path("mobilenetv2-7.onnx")?
            .with_input_fact(0, f32.fact(&[1, 3, 224, 224]).into())?
            .into_decluttered()?
```

## Model import pipeline

Once a model is loaded, tract default behaviour is to call the the `dump` subcommand, so the
previous example is equivalent to:

```bash
tract mobilenetv2-7.onnx -i 1,3,224,224,f32 dump
```

The displayed form is `tract-opl` intermediate representation. It is *decluttered* of most
training artefacts, in a form meant to be simple to reason about and as stripped down as
possible.

This is not the "optimised" form: `tract-opl` form is meant to be platform independant, can
be serialized to nnef. The optimised form is just meant to be as fast as possible on a given
CPU.

The `.into_optimized()` transformation can be performed by passing `-O` to the command line.

```bash
tract -O mobilenetv2-7.onnx -i 1,3,224,224,f32 dump
```

Several other intermediate network "stages" can be reached by using `--pass XXX` instead of `-O`.
`--pass load` and `--pass analyse` are interesting as they can dump a network for which inputs are
unknown (maybe to try and figure out what they could be).

## Benching a network

We can get a reading of tract performance on a model by running the `bench` or`criterion`
subcommands.

```
tract -O mobilenetv2-7.onnx -i 1,3,224,224,f32 bench
tract -O mobilenetv2-7.onnx -i 1,3,224,224,f32 criterion
```

The first one is a simple bench runner customized for tract specific needs, the second one
uses the [criterion](https://docs.rs/criterion) crate.

## Profiling a network

Getting a raw performance number is a first step, but tract can also profile a network execution.
A goto command to get a first glimpse can be:

```
tract -O mobilenetv2-7.onnx -i 1,3,224,224,f32 dump --profile --cost
```

This will show running time for each operator, its relative weight. For some critical operations,
it will also give a number of arithmetic operations per seconds (typically Flops).

Note that
we only count item multiplications, whereas many projects in the HPC field count both
multiplications and additions. So for matrix multiplication, convolution and the like, you may need
to double tract Flops number before comparing with, say, BLAS implementations.

Please do not parse this output. At least use the `--json` output. We do not commit on its stability
but it's less susceptible to changes.

## Running a test case

`tract` command line can also be use to build test-case, either for non-regression insurance
of debugging purposes.

`--input-facts-from-bundle` takes a `.npz` file, and will set the input facts (dtype, shape) according to the tensors
in the npz file. This is useful when your model does not have any input type information embedded within it.

The `run` subcommand accepts an `--input-from-bundle` that also takes a `.npz` file, but it
will not set any input fact, it will only take the tensor values.
This will also supersede the `-i` option: we will take the input shapes and tensor
from the input itself.

The `run` subcommand also accepts an `--assert-output-bundle`. This time, the tensors names are
matched with the model output names. `tract` will run over the input and check that its finding
are the same to the expected output (with some leeway for rounding differences).

[Example here](/onnx/test_cases/qtanh_1) for a quantized tanh in onnx.

```sh
tract model.onnx -O run --input-from-bundle io.npz --assert-output-bundle io.npz
```

If we want to make sure we actually check something, `-v` can help:

```
tract -v model.onnx -O run --input-from-bundle io.npz --assert-output-bundle io.npz
```

The log displays "Checked output #0, ok." (among other information).

[generate_io.py here](/onnx/test_cases/transformer-mlm/generate_io.py) contains an example building a
testcase for a BERT model from huggingface for inspiration.
