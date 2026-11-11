#!/usr/bin/env python3
"""Generate mixed negative/positive Pad cases with constant, reflect and axes inputs."""
from pathlib import Path
import onnx
from onnx import helper, TensorProto
for name,shape,pads,axes,mode,out_shape in [
    ("constant",[4],[-1,1],None,"constant",[4]),
    ("reflect",[4],[-1,2],None,"reflect",[5]),
    ("axes",[2,3],[-1,1],[1],"constant",[2,3]),
]:
    constants=[helper.make_tensor("pads",TensorProto.INT64,[len(pads)],pads)]
    inputs=["x","pads"]
    if axes is not None:
        constants.append(helper.make_tensor("axes",TensorProto.INT64,[len(axes)],axes))
        inputs += ["","axes"]
    model=helper.make_model(helper.make_graph(
        [helper.make_node("Pad",inputs,["y"],mode=mode)],name,
        [helper.make_tensor_value_info("x",TensorProto.FLOAT,shape)],
        [helper.make_tensor_value_info("y",TensorProto.FLOAT,out_shape)],constants),
        opset_imports=[helper.make_opsetid("",18)],ir_version=10)
    onnx.checker.check_model(model,full_check=True)
    onnx.save(model,Path(__file__).resolve().parent/(name+".onnx"))
