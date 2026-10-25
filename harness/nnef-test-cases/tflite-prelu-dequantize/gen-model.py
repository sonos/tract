#!/usr/bin/env python3
"""Generate model.tflite and io.npz for the tflite PRELU / DEQUANTIZE case.

A float32 Conv2D followed by a per-channel PReLU, converted with float16
weight quantization: the converter stores the kernel, the bias and the PReLU
alpha as FLOAT16 constants behind DEQUANTIZE ops. The reference output comes
from the TFLite interpreter.

Needs tensorflow (2.21 was used).
"""

import numpy as np
import tensorflow as tf
import keras

rng = np.random.default_rng(7)

inp = keras.Input((6, 6, 3), batch_size=1, name="x")
y = keras.layers.Conv2D(4, 3, padding="same", name="conv")(inp)
y = keras.layers.PReLU(shared_axes=[1, 2], name="prelu")(y)
model = keras.Model(inp, y)

for name in ("conv", "prelu"):
    layer = model.get_layer(name)
    layer.set_weights(
        [rng.uniform(-1, 1, w.shape).astype(np.float32) for w in layer.get_weights()]
    )

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
flatbuffer = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(flatbuffer)

x = rng.standard_normal((1, 6, 6, 3)).astype(np.float32)
interpreter = tf.lite.Interpreter(model_content=flatbuffer)
interpreter.allocate_tensors()
(i,) = interpreter.get_input_details()
(o,) = interpreter.get_output_details()
interpreter.set_tensor(i["index"], x)
interpreter.invoke()
output = interpreter.get_tensor(o["index"])

np.savez("io.npz", **{i["name"]: x, o["name"]: output})
