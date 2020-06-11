import onnxruntime
import numpy as np

n_pad = 5
texts = [
    "Das ist ein Test. Das ist noch ein Test.",
    "Das ist ein weiterer Test. Und ein zweiter Satz.",
]
boundaries = [[17, 39], [26, 47]]

# pad text by zeros and encode as utf-8 bytes
encoded = [[0] * n_pad + list(x.encode("utf-8")) + [0] * n_pad for x in texts]

# pda to same length with zeros at the end
max_length = max(len(x) for x in encoded)
padded = [np.pad(x, ((0, max(max_length - len(x), 0)),)) for x in encoded]
inputs = np.stack(padded, 0).astype(np.uint8)

sess = onnxruntime.InferenceSession("model.onnx")
input_name = sess.get_inputs()[0].name

# shape batch x len x 2
# value 1 in last dimension are token boundaries, value 0 are sentence boundaries
# outputs are logits, in practice there would be a sigmoid afterwards
outputs = sess.run(None, {input_name: inputs})[0]

assert len(outputs) == len(texts)

for i in range(len(outputs)):
    assert (np.where(outputs[i, :, 0] > 0)[0] - n_pad).tolist() == boundaries[i]

np.savez_compressed(open("io.npz", "wb"), input=inputs, Add_26=outputs)
