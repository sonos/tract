import numpy as np
from sklearn import datasets
from lightgbm.sklearn import LGBMClassifier
from hummingbird.ml import convert
import onnxruntime
import torch

x, y = datasets.load_wine(return_X_y=True)
x = x.astype(np.float32)

model = LGBMClassifier(n_estimators=5)
model.fit(x, y)
preds = model.predict_proba(x)

pytorch_model = convert(model, "pytorch")

torch.onnx.export(
    pytorch_model.model,
    (torch.from_numpy(x)),
    "model.onnx",
    input_names=["input"],
    output_names=["output", "probabilities"],
    dynamic_axes={
        "input": {0: "batch"},
        "output": {0: "batch"},
        "probabilities": {0: "batch"},
    },
)

np.savez_compressed(
    open("io.npz", "wb"), input=x[:1], probabilities=preds[:1],
)

# sanity check - onnxruntime inference

sess = onnxruntime.InferenceSession("model.onnx")
outputs = sess.run(["probabilities"], {"input": x[:1]})[0]

assert np.allclose(outputs, preds[:1])
