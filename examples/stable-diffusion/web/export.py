#!/usr/bin/env python3
"""SD 1.5's text encoder, UNet and VAE decoder as fp16 ONNX graphs that carry no weights: every
parameter is an external initializer {location, offset, length} into Hugging Face's own fp16
safetensors file, under its Hugging Face name. Only exporter-made constants (scalars, masks,
shape vectors) stay embedded.

Each graph is written next to the safetensors file it reads (<snapshot>/<component>/<name>.onnx),
where the external references resolve. export.sh then turns them into the NNEF tars the demo
loads.

Usage: export.py <hf snapshot dir>
"""

import json
import os
import struct
import sys
import tempfile

import numpy as np
import onnx
import onnx.inliner
import torch

HF = sys.argv[1]
MAX_EMBEDDED_BYTES = 4 << 20


def safetensors_index(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return 8 + n, header


ST_DTYPES = {"F16": onnx.TensorProto.FLOAT16, "F32": onnx.TensorProto.FLOAT, "I64": onnx.TensorProto.INT64}


def split_sequences(model):
    """Rewrites SplitToSequence (a constant chunk size) read back only by SequenceAt (constant
    indices), which is what torch.chunk exports to, as one Split. Fails on any other use."""
    g = model.graph
    prod = {o: n for n in g.node for o in n.output}
    users = {}
    for n in g.node:
        for i in n.input:
            users.setdefault(i, []).append(n)
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in g.value_info}

    def const(name):
        n = prod[name]
        assert n.op_type == "Constant", f"{name} is not a Constant"
        return int(onnx.numpy_helper.to_array(n.attribute[0].t))

    nodes, rewritten = [], 0
    for n in g.node:
        if n.op_type == "SequenceAt" and prod.get(n.input[0]) is not None \
                and prod[n.input[0]].op_type == "SplitToSequence":
            continue
        if n.op_type != "SplitToSequence":
            nodes.append(n)
            continue
        axis = next(a.i for a in n.attribute if a.name == "axis")
        chunk = const(n.input[1])
        dim = shapes[n.input[0]][axis]
        assert dim % chunk == 0, f"{n.name}: {dim} does not split in chunks of {chunk}"
        reads = users[n.output[0]]
        assert all(u.op_type == "SequenceAt" for u in reads), f"{n.name}: read by more than SequenceAt"
        outputs = [f"{n.name}_unused_{k}" for k in range(dim // chunk)]
        for u in reads:
            outputs[const(u.input[1])] = u.output[0]
        sizes = f"{n.name}_sizes"
        nodes.append(onnx.helper.make_node("Constant", [], [sizes], value=onnx.numpy_helper.from_array(
            np.array([chunk] * (dim // chunk), dtype=np.int64))))
        nodes.append(onnx.helper.make_node("Split", [n.input[0], sizes], outputs, name=n.name, axis=axis))
        rewritten += 1
    del g.node[:]
    g.node.extend(nodes)
    return rewritten


def export(module, args, input_names, output_names, component, weights_file, onnx_name, prefix=""):
    out_dir = os.path.join(HF, component)
    weights_path = os.path.join(out_dir, weights_file)
    data_start, header = safetensors_index(weights_path)
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "raw.onnx")
        torch.onnx.export(
            module, args, raw, input_names=input_names, output_names=output_names,
            dynamo=True, optimize=False, external_data=True,
        )
        model = onnx.load(raw, load_external_data=True)
    # The exporter wraps some ops as onnxscript local functions; inline them into plain ops.
    model = onnx.inliner.inline_local_functions(model)
    print(f"{component}: {split_sequences(model)} SplitToSequence rewritten as Split")

    blob = open(weights_path, "rb")
    mapped, embedded = 0, 0
    renames = {}
    for init in model.graph.initializer:
        meta = header.get(init.name.removeprefix(prefix))
        if meta is None:
            embedded += onnx.numpy_helper.to_array(init).nbytes
            continue
        assert list(init.dims) == meta["shape"], (init.name, list(init.dims), meta["shape"])
        assert init.data_type == ST_DTYPES[meta["dtype"]], (init.name, init.data_type, meta["dtype"])
        start, end = meta["data_offsets"]
        blob.seek(data_start + start)
        expected = blob.read(end - start)
        actual = onnx.numpy_helper.to_array(init).tobytes()
        assert actual == expected, f"{init.name}: exported bytes differ from the safetensors slice"
        hf_name = init.name.removeprefix(prefix)
        if hf_name != init.name:
            renames[init.name] = hf_name
            init.name = hf_name
        init.ClearField("raw_data")
        init.ClearField("float_data")
        init.ClearField("int64_data")
        init.data_location = onnx.TensorProto.EXTERNAL
        del init.external_data[:]
        for k, v in (("location", weights_file), ("offset", str(data_start + start)), ("length", str(end - start))):
            entry = init.external_data.add()
            entry.key, entry.value = k, v
        mapped += 1
    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name in renames:
                node.input[i] = renames[name]
    assert embedded <= MAX_EMBEDDED_BYTES, f"{component}: {embedded} bytes of weights left embedded"
    dst = os.path.join(out_dir, onnx_name)
    onnx.save(model, dst)
    print(f"{component}: {mapped} initializers into {weights_file}, {embedded} bytes embedded, "
          f"graph {os.path.getsize(dst)} bytes -> {dst}")


def main():
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel

    dt = torch.float16

    text_encoder = CLIPTextModel.from_pretrained(HF, subfolder="text_encoder", variant="fp16", torch_dtype=dt).eval()

    class TextEncoder(torch.nn.Module):
        def __init__(self, te):
            super().__init__()
            self.te = te

        def forward(self, input_ids):
            return self.te(input_ids).last_hidden_state

    export(TextEncoder(text_encoder), (torch.zeros(2, 77, dtype=torch.int64),), ["input_ids"],
           ["last_hidden_state"], "text_encoder", "model.fp16.safetensors", "text_encoder.onnx", "te.")

    unet = UNet2DConditionModel.from_pretrained(HF, subfolder="unet", variant="fp16", torch_dtype=dt).eval()

    class Unet(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet

        def forward(self, sample, timestep, encoder_hidden_states):
            return self.unet(sample, timestep, encoder_hidden_states).sample

    export(Unet(unet), (torch.zeros(2, 4, 64, 64, dtype=dt), torch.zeros(2, dtype=torch.int64),
                        torch.zeros(2, 77, 768, dtype=dt)),
           ["sample", "timestep", "encoder_hidden_states"], ["noise_pred"],
           "unet", "diffusion_pytorch_model.fp16.safetensors", "unet.onnx", "unet.")

    vae = AutoencoderKL.from_pretrained(HF, subfolder="vae", variant="fp16", torch_dtype=dt).eval()

    class VaeDecoder(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.post_quant_conv = vae.post_quant_conv
            self.decoder = vae.decoder

        def forward(self, latent):
            return self.decoder(self.post_quant_conv(latent))

    export(VaeDecoder(vae), (torch.zeros(1, 4, 64, 64, dtype=dt),), ["latent"], ["image"],
           "vae", "diffusion_pytorch_model.fp16.safetensors", "vae_decoder.onnx")


if __name__ == "__main__":
    with torch.no_grad():
        main()
