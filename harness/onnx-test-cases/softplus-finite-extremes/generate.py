#!/usr/bin/env python3
"""Generate Softplus models for the supported floating-point input types."""
from pathlib import Path
import onnx
from onnx import helper, TensorProto
for name, dtype in [("f32", TensorProto.FLOAT), ("f16", TensorProto.FLOAT16)]:
    model=helper.make_model(helper.make_graph(
        [helper.make_node("Softplus",["x"],["y"])],name,
        [helper.make_tensor_value_info("x",dtype,[7])],
        [helper.make_tensor_value_info("y",dtype,[7])]),
        opset_imports=[helper.make_opsetid("",13)],ir_version=10)
    onnx.checker.check_model(model,full_check=True)
    onnx.save(model,Path(__file__).resolve().parent/(name+".onnx"))
