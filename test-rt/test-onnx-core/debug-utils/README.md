

cargo run -- ../../../.cached/onnx/onnx/backend/test/data/real/test_inception_v1/inception_v1/model.onnx inception_v1_all_outputs.onnx

virtualenv -p python3 ort
source ./ort/bin/activate
pip install numpy onnx onnxruntime

python ./save_all.py inception_v1_all_outputs.onnx inception_v1_all_outputs.npz ../../../.cached/onnx/onnx/backend/test/data/real/test_inception_v1/inception_v1/test_data_0.npz
