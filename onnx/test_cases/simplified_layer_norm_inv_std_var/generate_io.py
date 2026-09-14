"""SimplifiedLayerNormalization (onnxruntime contrib op) with its optional inv_std_var output.

inv_std_var is 1 / sqrt(mean(x^2) + epsilon) over the normalised axes, which are kept with
size 1. The input has a non-zero mean and the scale is non-unit. Expected outputs come
from onnxruntime.
"""
import numpy as np, onnx, onnxruntime
from onnx import helper, TensorProto

node = helper.make_node("SimplifiedLayerNormalization", ["x", "scale"], ["y", "inv_std_var"],
                        axis=-1, epsilon=1e-5)
vi = helper.make_tensor_value_info
graph = helper.make_graph([node], "simplified_layer_norm_inv_std_var",
                          [vi("x", TensorProto.FLOAT, [3, 8]), vi("scale", TensorProto.FLOAT, [8])],
                          [vi("y", TensorProto.FLOAT, [3, 8]),
                           vi("inv_std_var", TensorProto.FLOAT, [3, 1])])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17),
                                                helper.make_opsetid("com.microsoft", 1)])
model.ir_version = 9
onnx.save(model, "model.onnx")

rng = np.random.RandomState(20260914)
inputs = dict(x=(rng.randn(3, 8) * 2 + 1).astype(np.float32),
              scale=(rng.rand(8) + 0.5).astype(np.float32))
sess = onnxruntime.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
y, inv_std_var = sess.run(None, inputs)
np.savez_compressed("io.npz", y=y, inv_std_var=inv_std_var, **inputs)
