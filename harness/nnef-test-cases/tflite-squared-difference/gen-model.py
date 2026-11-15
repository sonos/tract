#!/usr/bin/env python3
"""Generate model.tflite and io.npz for the tflite SQUARED_DIFFERENCE case.

Two inputs of different rank, so the op exercises the deserializer's cast and
rank broadcast: z = (x - y)^2 with x [1, 4, 3] and y [1, 1, 3]. TensorFlow
lowered squared_difference to SUB+SQUARE in older versions; the generator
asserts the builtin op is what the flatbuffer carries. The reference output
comes from the TFLite interpreter.

Needs tensorflow (2.21 was used).
"""

import numpy as np
import tensorflow as tf

rng = np.random.default_rng(11)
x = rng.standard_normal((1, 4, 3)).astype(np.float32)
y = (rng.standard_normal((1, 1, 3)) * 0.5).astype(np.float32)


@tf.function(
    input_signature=[
        tf.TensorSpec([1, 4, 3], tf.float32, name="x"),
        tf.TensorSpec([1, 1, 3], tf.float32, name="y"),
    ]
)
def squared_difference(x, y):
    return tf.math.squared_difference(x, y)


converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [squared_difference.get_concrete_function()], squared_difference
)
flatbuffer = converter.convert()

interpreter = tf.lite.Interpreter(model_content=flatbuffer)
interpreter.allocate_tensors()
ops = [d["op_name"] for d in interpreter._get_ops_details()]
assert "SQUARED_DIFFERENCE" in ops, ops

with open("model.tflite", "wb") as f:
    f.write(flatbuffer)

(i_x, i_y) = interpreter.get_input_details()
(o,) = interpreter.get_output_details()
interpreter.set_tensor(i_x["index"], x)
interpreter.set_tensor(i_y["index"], y)
interpreter.invoke()
output = interpreter.get_tensor(o["index"])

np.savez(
    "io.npz",
    **{i_x["name"]: x, i_y["name"]: y, o["name"]: output},
)
