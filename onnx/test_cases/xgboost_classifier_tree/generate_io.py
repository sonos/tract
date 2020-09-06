import numpy as np
from sklearn import datasets
from xgboost.sklearn import XGBClassifier
from onnxmltools.convert.common import data_types
import onnxmltools
import onnx
import onnxruntime

x, y = datasets.load_wine(return_X_y=True)
x = x.astype(np.float32)

model = XGBClassifier(n_estimators=10)
model.fit(x, y)
preds = model.predict_proba(x)

onnx_model = onnxmltools.convert_xgboost(
    model, initial_types=[("input", data_types.FloatTensorType([None, x.shape[1]]))],
)

onnx.save(onnx_model, "model.onnx")

np.savez_compressed(
    open("io.npz", "wb"), input=x[:1], probabilities=preds[:1],
)

# sanity check - onnxruntime inference

sess = onnxruntime.InferenceSession("model.onnx")
outputs = sess.run(["probabilities"], {"input": x[:1]})[0]

assert np.allclose(outputs, preds[:1])
