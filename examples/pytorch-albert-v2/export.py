# -*- coding: utf-8 -*-

import os

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from torch.export import Dim

model_name = "albert-base-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

text = "Paris is the [MASK] of France."
tokenizer_output = tokenizer(text, return_tensors="pt")

input_ids = tokenizer_output["input_ids"]
attention_mask = tokenizer_output["attention_mask"]
token_type_ids = tokenizer_output["token_type_ids"]

batch = Dim("batch")
seq = Dim("seq")

output_dir = "./albert"
os.makedirs(output_dir, exist_ok=True)
torch.onnx.export(
    model,
    (input_ids, attention_mask, token_type_ids),
    os.path.join(output_dir, "model.onnx"),
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["logits"],
    dynamic_shapes={
        "input_ids": (batch, seq),
        "attention_mask": (batch, seq),
        "token_type_ids": (batch, seq),
    }, 
)

tokenizer.save_pretrained(output_dir)
