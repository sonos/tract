import numpy as np
from sklearn import datasets
from xgboost.sklearn import XGBRegressor
from onnxmltools.convert.common import data_types
import onnxmltools
import onnx
import onnxruntime

x, y = datasets.load_wine(return_X_y=True)
x = x.astype(np.float32)

model = XGBRegressor(n_estimators=10)
model.fit(x, y)
preds = model.predict(x)

onnx_model = onnxmltools.convert_xgboost(
    model, initial_types=[("input", data_types.FloatTensorType([None, x.shape[1]]))],
)

onnx.save(onnx_model, "model.onnx")

np.savez_compressed(
    open("io.npz", "wb"), input=x[:1], variable=preds[:1],
)

# sanity check - onnxruntime inference

sess = onnxruntime.InferenceSession("model.onnx")
outputs = sess.run(None, {"input": x[:1]})[0][:, 0]

assert np.allclose(outputs, preds[:1])
