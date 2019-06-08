import sys
import numpy
import onnxruntime as rt

model = sys.argv[1]
print(model)
sess = rt.InferenceSession(model)

input_data = sys.argv[2]
input_data = numpy.load(input_data)

input_data = numpy.squeeze(input_data["inputs"], 0)

input_name = sess.get_inputs()[0].name
pred_onnx = sess.run(None, {input_name: input_data})

outputs = {input_name: input_data}

for ix, output in enumerate(sess.get_outputs()):
    print(output.name)
    outputs[output.name] = pred_onnx[ix]

numpy.savez(sys.argv[3], **outputs)
