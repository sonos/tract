import sys
import numpy
import onnx
from onnx import numpy_helper
import onnxruntime as rt

model = sys.argv[1]
output_name = sys.argv[2]

print("model: " + model)
print("output: " + output_name)
sess = rt.InferenceSession(model)

inputs = {}
for i in range(3, len(sys.argv)):
    input_data = sys.argv[i]
    if input_data.endswith(".npz"):
        input_data = numpy.load(input_data)
        input_data = numpy.squeeze(input_data, 0)
        name = sess.get_inputs()[i-3].name
    elif input_data.endswith(".pb"):
        new_tensor = onnx.TensorProto()
        with open(input_data, 'rb') as f:
            new_tensor.ParseFromString(f.read())
        name = new_tensor.name
        input_data = numpy_helper.to_array(new_tensor)
    inputs[name] = input_data

outputs = inputs.copy()
pred_onnx = sess.run(None, inputs)

for ix, output in enumerate(sess.get_outputs()):
    print(output.name)
    outputs[output.name] = pred_onnx[ix]

numpy.savez(output_name, **outputs)
