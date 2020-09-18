import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
import onnxruntime

model_name = "distilbert-base-uncased"
model_path = "model.onnx"

text = "tract is a machine [MASK] library."
filler = "learning"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

encoded = tokenizer.encode_plus(text)
mask_idx = encoded["input_ids"].index(tokenizer.mask_token_id)

input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long)
attention_mask = torch.tensor([encoded["attention_mask"]], dtype=torch.long)

torch.onnx.export(
    model,
    (input_ids, attention_mask),
    model_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["output"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "output": {0: "batch", 1: "seq"},
    },
)

sess = onnxruntime.InferenceSession(model_path)

outputs = sess.run(
    None, {"input_ids": input_ids.numpy(), "attention_mask": attention_mask.numpy()}
)[0]
assert tokenizer.convert_ids_to_tokens(int(np.argmax(outputs[0, mask_idx]))) == filler

np.savez_compressed(
    open("io.npz", "wb"),
    input_ids=input_ids.numpy(),
    attention_mask=attention_mask.numpy(),
    output=outputs,
)
