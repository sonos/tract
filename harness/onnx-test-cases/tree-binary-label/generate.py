"""Generate binary classifiers with a single score channel and non-index labels."""
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def write(name, weights, transform):
    node = helper.make_node(
        "TreeEnsembleClassifier", ["x"], ["label", "scores"], domain="ai.onnx.ml",
        classlabels_int64s=[11, 21], class_ids=[0, 0], class_nodeids=[1, 2],
        class_treeids=[0, 0], class_weights=weights, nodes_treeids=[0, 0, 0],
        nodes_nodeids=[0, 1, 2], nodes_featureids=[0, 0, 0],
        nodes_values=[0.0, 0.0, 0.0], nodes_modes=["BRANCH_LEQ", "LEAF", "LEAF"],
        nodes_truenodeids=[1, 0, 0], nodes_falsenodeids=[2, 0, 0],
        post_transform=transform,
    )
    value = helper.make_tensor_value_info
    graph = helper.make_graph(
        [node], name, [value("x", TensorProto.FLOAT, [2, 1])],
        [value("label", TensorProto.INT64, [2]), value("scores", TensorProto.FLOAT, [2, 2])],
    )
    model = helper.make_model(graph, ir_version=9, opset_imports=[
        helper.make_opsetid("", 18), helper.make_opsetid("ai.onnx.ml", 3)
    ])
    onnx.checker.check_model(model)
    onnx.save(model, Path(__file__).with_name(f"{name}.onnx"))


write("probability", [0.5, 0.75], "NONE")
write("signed", [-0.2, 0.1], "NONE")
write("post_transform", [0.2, 0.8], "LOGISTIC")
