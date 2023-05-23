# tract-tflite

unimplemented, sausage is being made. If you want to help feel free to open a PR.

## Notes and Relevant Links

[link to the related issue](https://github.com/sonos/tract/issues/1086)

right now the main task is to create a typedmodel from the model from the flatbuffer table. I'll add more here as progress is made.

### Operators

- the list of builtin Operators can be found in the [generated tflite schema](./src/tflite_generated.rs) around line 443 in the const array `ENUM_VALUES_BUILTIN_OPERATOR: [BuiltinOperator; 162]`.
- the offical docs on supported supset of tensorflow operators in [TFLite](https://www.tensorflow.org/lite/guide/op_select_allowlist)
- the [tflite c code](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/c)

### scratchpad

I created a [repository for the sole purpose of poking around with tflite models](https://github.com/skewballfox/tflite_scratch), if you would like to add a model for testing please put it inside test data, and add any test input to lfs. If you write some utility that would be useful for others contributers, feel free to add it. Otherwise just clone it and forget it, it's just trow-away code.
