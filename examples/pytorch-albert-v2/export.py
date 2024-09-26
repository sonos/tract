# -*- coding: utf-8 -*-

import os

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_name = "albert-base-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

text = "Paris is the [MASK] of France."
tokenizer_output = tokenizer(text, return_tensors="pt")

input_ids = tokenizer_output["input_ids"]
attention_mask = tokenizer_output["attention_mask"]
token_type_ids = tokenizer_output["token_type_ids"]

dynamic_axes = {
    0: "batch",
    1: "seq",
}

output_dir = "./albert"
os.makedirs(output_dir, exist_ok=True)
torch.onnx.export(
    model,
    (input_ids, attention_mask, token_type_ids),
    os.path.join(output_dir, "model.onnx"),
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": dynamic_axes,
        "attention_mask": dynamic_axes,
        "token_type_ids": dynamic_axes,
        "logits": dynamic_axes,
    },
    opset_version=14,
)

tokenizer.save_pretrained(output_dir)
