import os
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

known = {}
for i in range(3, len(sys.argv)):
    input_data = sys.argv[i]
    if input_data.endswith(".npz"):
        tensors = numpy.load(input_data)
        for name, array in tensors.items():
            known[name] = array
        name = sess.get_inputs()[i-3].name
    elif input_data.endswith(".pb"):
        new_tensor = onnx.TensorProto()
        with open(input_data, 'rb') as f:
            new_tensor.ParseFromString(f.read())
        name = new_tensor.name
        input_data = numpy_helper.to_array(new_tensor)

print("known: ", known)

inputs = {}
for input in sess.get_inputs():
    inputs[input.name] = known[input.name]

outputs = inputs.copy()
pred_onnx = sess.run(None, inputs)

for ix, output in enumerate(sess.get_outputs()):
    outputs[output.name] = pred_onnx[ix]

print("computed: ", outputs.keys())

for name, array in outputs.items():
    print(name)
    print(array)

os.mkdir(output_name)

for name in outputs:
    value = numpy_helper.from_array(outputs[name], name=name)
    with open(output_name + "/" + name + ".pb", 'wb') as f:
        f.write(value.SerializeToString())
