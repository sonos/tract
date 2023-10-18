# tract-tflite

unimplemented, sausage is being made. If you want to help feel free to open a PR.

## Notes and Relevant Links

[link to the tflite c api](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/c)

[link to the related issue](https://github.com/sonos/tract/issues/1086)

The generated code handles creating a model from a flatbuffer table. Right now the main task (as far as I understand) is to start adding the code to build a Tract Model from the ModelBuffer.

So the modelBuffer(the model read from a flatbuffer file) has a few components (with associated functions) worth looking at: operator_codes, subgraphs, and then buffers.

- subgraphs are likely the primary thing needed to create a tract model
  - composed of tensors, inputs,outputs, operators, and a name
  - input and output are fairly small vectors, I suspect they may be indices
- buffers are sometimes empty (why?)

## Metadata

- [tensorflow docs on metadate, has information on subgraphs as well](https://www.tensorflow.org/lite/models/convert/metadata)

## Tensors

- probably need to convert from the generated datatypes to Tract's [DatumType](https://github.com/skewballfox/tract/blob/300db595a1ffe3088658643b694b41aaac71ee76/data/src/datum.rs#L121). it's in the toplevel data crate.
  - this is part of the depenendency tract-core
- [SO: what a variant tensor?](https://stackoverflow.com/questions/58899763/what-is-a-dt-variant-tensor)

### Operators

- the list of builtin Operators can be found in the [generated tflite schema](./src/tflite_generated.rs) around line 443 in the const array `ENUM_VALUES_BUILTIN_OPERATOR: [BuiltinOperator; 162]`.
- the official docs on supported supset of tensorflow operators in [TFLite](https://www.tensorflow.org/lite/guide/op_select_allowlist)
- the [tflite c code](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/c)

### Subgraphs

Right now, I'm testing with a specific model under test data, so this might not generalize to other models. If you open the model in [netron](netron.app), you'll find 3 separate graphs: main, sequential/net/while_cond, and sequential/net/while_body.

In the main graph, node 10 is just listed as while, but it's actually composed of the other subgraphs.

### scratchpad

I created a [repository for the sole purpose of poking around with tflite models](https://github.com/skewballfox/tflite_scratch), if you would like to add a model for testing please put it inside test data, and add any test input to lfs. If you write some utility that would be useful for others contributers, feel free to add it. Otherwise just clone it and forget it, it's just trow-away code.
