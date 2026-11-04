#!/usr/bin/env python3
"""Export Bonsai Image (ternary Flux2 Klein 4B) components to ONNX."""

import os
from pathlib import Path

import numpy as np
import torch

ASSETS = Path(os.environ.get("ASSETS", "assets"))
HEIGHT, WIDTH = map(int, os.environ.get("SIZE", "256x256").split("x"))
TEXT_LEN = int(os.environ.get("TEXT_LEN", 64))
BLOCK = 128
TEXT_LAYERS = (9, 18, 27)
MODEL_ID = "prism-ml/bonsai-image-ternary-4B-unpacked"


class VaeDecoder(torch.nn.Module):
    """Packed latent (1, 128, HEIGHT/16, WIDTH/16) -> image (1, 3, HEIGHT, WIDTH) in [-1, 1].

    Folds the pipeline's batch-norm denormalisation and 2x2 unpatchify into the graph.
    """

    def __init__(self, vae):
        super().__init__()
        eps = vae.config.batch_norm_eps
        self.register_buffer("bn_mean", vae.bn.running_mean.view(1, -1, 1, 1).clone())
        self.register_buffer("bn_std", torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + eps))
        self.post_quant_conv = vae.post_quant_conv
        self.decoder = vae.decoder

    def forward(self, latent):
        latent = latent * self.bn_std + self.bn_mean
        b, c, h, w = latent.shape
        latent = latent.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
        latent = latent.reshape(b, c // 4, h * 2, w * 2)
        return self.decoder(self.post_quant_conv(latent))


def export_vae():
    from diffusers import AutoencoderKLFlux2

    print(f"Loading {MODEL_ID} vae...")
    vae = AutoencoderKLFlux2.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
    vae.eval()
    model = VaeDecoder(vae).eval()
    h, w = HEIGHT // 16, WIDTH // 16
    latent = torch.randn(1, 128, h, w, generator=torch.Generator().manual_seed(0))
    path = ASSETS / "vae_decoder.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model,
            (latent,),
            str(path),
            input_names=["latent"],
            output_names=["image"],
            opset_version=17,
            dynamo=False,
        )
        image = model(latent)
    np.savez(ASSETS / "vae_decoder.io.npz", latent=latent.numpy(), image=image.numpy())
    tokens = latent.reshape(1, 128, h * w).permute(0, 2, 1).contiguous()
    np.save(ASSETS / "vae_tokens.npy", tokens.numpy())
    np.save(ASSETS / "vae_image.npy", image.numpy())
    print(f"  Exported to {path}")


class _NBitsMatMul(torch.autograd.Function):
    """Traces to a com.microsoft MatMulNBits holding a placeholder weight."""

    @staticmethod
    def forward(ctx, x, n, k, bits, name):
        return x.new_zeros(*x.shape[:-1], n)

    @staticmethod
    def symbolic(g, x, n, k, bits, name):
        stub = g.op("Constant", value_t=torch.zeros(1, dtype=torch.uint8))
        block = BLOCK if bits == 2 else 32
        inputs = [x, stub, stub] + ([stub] if bits == 2 else [])
        return g.op(
            "com.microsoft::MatMulNBits", *inputs,
            K_i=k, N_i=n, bits_i=bits, block_size_i=block, weight_s=name,
        )


class NBitsLinear(torch.nn.Module):
    def __init__(self, name, n, k, bits):
        super().__init__()
        self.name, self.n, self.k, self.bits = name, n, k, bits

    def forward(self, x):
        return _NBitsMatMul.apply(x, self.n, self.k, self.bits, self.name)


class ExplicitRmsNorm(torch.nn.Module):
    """RMSNorm spelled in plain ops, which the ONNX exporter has no symbolic for."""

    def __init__(self, norm):
        super().__init__()
        self.eps = norm.eps
        self.weight = norm.weight

    def forward(self, x):
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return y if self.weight is None else y * self.weight


class Transformer(torch.nn.Module):
    """hidden (1, N, 128), enc (1, TEXT_LEN, 7680), t (1) -> velocity (1, N, 128).

    Position ids are fixed by HEIGHT, WIDTH and TEXT_LEN, so they are constants of the graph.
    """

    def __init__(self, transformer, img_ids, txt_ids):
        super().__init__()
        self.transformer = transformer
        self.register_buffer("img_ids", img_ids)
        self.register_buffer("txt_ids", txt_ids)

    def forward(self, hidden, enc, t):
        return self.transformer(
            hidden_states=hidden,
            encoder_hidden_states=enc,
            timestep=t,
            img_ids=self.img_ids,
            txt_ids=self.txt_ids,
            guidance=None,
            return_dict=False,
        )[0]


def transformer_inputs():
    from diffusers import Flux2KleinPipeline

    h, w = HEIGHT // 16, WIDTH // 16
    g = torch.Generator().manual_seed(0)
    hidden = torch.randn(1, h * w, 128, generator=g)
    enc = torch.randn(1, TEXT_LEN, 7680, generator=g)
    t = torch.tensor([0.75])
    img_ids = Flux2KleinPipeline._prepare_latent_ids(torch.zeros(1, 128, h, w))[0]
    txt_ids = Flux2KleinPipeline._prepare_text_ids(enc)[0]
    return hidden, enc, t, img_ids.float(), txt_ids.float()


def is_ternary(weights, key):
    w = weights.get_tensor(key).float().numpy()
    n, k = w.shape
    if k % BLOCK:
        return False
    mag = np.abs(w).reshape(n, k // BLOCK, BLOCK)
    return bool(np.all((mag == 0) | (mag == mag.max(axis=-1, keepdims=True))))


def pack_ternary(weights, key):
    """Q2_0_T codes (-1 -> 0, 0 -> 1, +1 -> 2) and one scale per row and block."""
    w = weights.get_tensor(key).float().numpy()
    n, k = w.shape
    blocks = k // BLOCK
    scales = np.abs(w).reshape(n, blocks, BLOCK).max(axis=-1).astype(np.float32)
    c = (np.sign(w).astype(np.int8) + 1).astype(np.uint8).reshape(n, blocks, BLOCK)
    packed = c[..., 0::4] | c[..., 1::4] << 2 | c[..., 2::4] << 4 | c[..., 3::4] << 6
    return packed, scales


def pack_q4(weights, key):
    """Q4_0 codes `(code - 8) * scale`, one scale per row and block of 32."""
    w = weights.get_tensor(key).float().numpy()
    n, k = w.shape
    blocks = k // 32
    v = w.reshape(n, blocks, 32)
    scales = (np.abs(v).max(axis=-1) / 7).astype(np.float32)
    safe = np.where(scales == 0, 1, scales)[..., None]
    c = (np.clip(np.rint(v / safe), -8, 7) + 8).astype(np.uint8)
    return c[..., 0::2] | c[..., 1::2] << 4, scales


def attach_packed_weights(path, weights, output_shape):
    """Swaps MatMulNBits placeholders for packed weights in an external data file."""
    import onnx
    from onnx import TensorProto, numpy_helper

    model = onnx.load(path)
    data_name = path.name + ".data"
    written = 0

    def external(name, array, data_type, out):
        nonlocal written
        raw = array.tobytes()
        tensor = TensorProto(name=name, data_type=data_type, dims=array.shape)
        tensor.data_location = TensorProto.EXTERNAL
        for key, value in (("location", data_name), ("offset", written), ("length", len(raw))):
            entry = tensor.external_data.add()
            entry.key, entry.value = key, str(value)
        out.write(raw)
        written += len(raw)
        return tensor

    with open(path.parent / data_name, "wb") as out:
        for node in model.graph.node:
            if node.op_type != "MatMulNBits":
                continue
            attr = next(a for a in node.attribute if a.name == "weight")
            key = attr.s.decode() + ".weight"
            node.attribute.remove(attr)
            bits = next(a.i for a in node.attribute if a.name == "bits")
            packed, scales = (pack_ternary if bits == 2 else pack_q4)(weights, key)
            names = [f"{key}.codes", f"{key}.scales"]
            model.graph.initializer.extend(
                [
                    external(names[0], packed, TensorProto.UINT8, out),
                    external(names[1], scales.reshape(-1), TensorProto.FLOAT, out),
                ]
            )
            if bits == 2:
                n, blocks = scales.shape
                zero_points = np.full((n, -(-blocks // 4)), 0x55, dtype=np.uint8)
                names.append(f"{key}.zero_points")
                model.graph.initializer.append(
                    numpy_helper.from_array(zero_points.reshape(-1), names[2])
                )
            node.input[1:] = names
    used = {i for node in model.graph.node for i in node.input}
    keep = [n for n in model.graph.node if n.op_type != "Constant" or n.output[0] in used]
    del model.graph.node[:]
    model.graph.node.extend(keep)
    del model.graph.value_info[:]
    for i, dim in enumerate(model.graph.output[0].type.tensor_type.shape.dim):
        dim.Clear()
        dim.dim_value = output_shape[i]
    onnx.save(model, path)
    return written


def export_transformer():
    from diffusers import Flux2Transformer2DModel
    from safetensors import safe_open

    weights_path = Path(os.environ["HF_SNAPSHOT"]) / "transformer/diffusion_pytorch_model.safetensors"
    weights = safe_open(weights_path, "pt")
    config = Flux2Transformer2DModel.load_config(weights_path.parent)
    with torch.device("meta"):
        transformer = Flux2Transformer2DModel.from_config(config)
    plain = {}
    replaced = 0
    for name, module in list(transformer.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, torch.nn.RMSNorm):
                setattr(module, child_name, ExplicitRmsNorm(child))
            if not isinstance(child, torch.nn.Linear):
                continue
            full = f"{name}.{child_name}" if name else child_name
            assert child.bias is None, full
            if is_ternary(weights, full + ".weight"):
                setattr(module, child_name, NBitsLinear(full, child.out_features, child.in_features, 2))
                replaced += 1
    for key in weights.keys():
        if key in transformer.state_dict():
            plain[key] = weights.get_tensor(key).float()
    print(f"  {replaced} ternary matmuls, {len(plain)} plain tensors")
    transformer.load_state_dict(plain, strict=True, assign=True)
    hidden, enc, t, img_ids, txt_ids = transformer_inputs()
    model = Transformer(transformer.eval(), img_ids, txt_ids).eval()
    path = ASSETS / "transformer.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model,
            (hidden, enc, t),
            str(path),
            input_names=["hidden", "enc", "t"],
            output_names=["velocity"],
            opset_version=18,
            custom_opsets={"com.microsoft": 1},
            dynamo=False,
        )
    written = attach_packed_weights(path, weights, hidden.shape)
    print(f"  Exported to {path}, {written / 1e9:.2f} GB of packed weights")


class LazyLinear(torch.nn.Module):
    """A bias-free linear that reads its weight per call, so references fit in memory."""

    def __init__(self, weights, key):
        super().__init__()
        self.weights, self.key = weights, key

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weights.get_tensor(self.key).float())


def transformer_reference():
    """Runs the original weights in f32 on the fixed inputs for tract to compare against."""
    from diffusers import Flux2Transformer2DModel
    from safetensors import safe_open

    weights_path = Path(os.environ["HF_SNAPSHOT"]) / "transformer/diffusion_pytorch_model.safetensors"
    weights = safe_open(weights_path, "pt")
    config = Flux2Transformer2DModel.load_config(weights_path.parent)
    with torch.device("meta"):
        transformer = Flux2Transformer2DModel.from_config(config)
    for name, module in list(transformer.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, torch.nn.Linear):
                full = f"{name}.{child_name}" if name else child_name
                setattr(module, child_name, LazyLinear(weights, full + ".weight"))
    plain = {k: weights.get_tensor(k).float() for k in transformer.state_dict()}
    transformer.load_state_dict(plain, strict=True, assign=True)
    hidden, enc, t, img_ids, txt_ids = transformer_inputs()
    model = Transformer(transformer.eval(), img_ids, txt_ids).eval()
    with torch.no_grad():
        velocity = model(hidden, enc, t)
    np.savez(
        ASSETS / "transformer.io.npz",
        hidden=hidden.numpy(),
        enc=enc.numpy(),
        t=t.numpy(),
        velocity=velocity.numpy(),
    )


class TextEncoder(torch.nn.Module):
    """input_ids, attention_mask (1, TEXT_LEN) -> embeddings (1, TEXT_LEN, 7680).

    The pipeline concatenates the outputs of layers 9, 18 and 27, so nothing later is kept.
    """

    def __init__(self, qwen):
        super().__init__()
        self.qwen = qwen

    def forward(self, input_ids, attention_mask):
        out = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        return torch.cat([out.hidden_states[k] for k in TEXT_LAYERS], dim=-1)


def text_inputs():
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(Path(os.environ["HF_SNAPSHOT"]) / "tokenizer/tokenizer.json"))
    text = (
        "<|im_start|>user\na photo of a bonsai tree<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    mask = [1] * len(ids) + [0] * (TEXT_LEN - len(ids))
    ids = ids + [tokenizer.token_to_id("<|endoftext|>")] * (TEXT_LEN - len(ids))
    return torch.tensor([ids]), torch.tensor([mask])


def open_text_weights():
    from safetensors import safe_open

    default = Path(os.environ["HF_SNAPSHOT"]) / "text_encoder"
    root = Path(os.environ.get("TEXT_ENCODER_DIR", default))
    return root, [safe_open(f, "pt") for f in sorted(root.glob("model-*.safetensors"))]


class ShardedWeights:
    def __init__(self, shards):
        self.where = {k: s for s in shards for k in s.keys()}

    def get_tensor(self, key):
        return self.where[key].get_tensor(key)


def build_text_encoder(weights, config, linear):
    from transformers import AutoModel

    config.num_hidden_layers = max(TEXT_LAYERS)
    if getattr(config, "layer_types", None):
        config.layer_types = config.layer_types[: config.num_hidden_layers]
    with torch.device("meta"):
        qwen = AutoModel.from_config(config, attn_implementation="eager")
    qwen.norm = torch.nn.Identity()
    qwen.rotary_emb = type(qwen.rotary_emb)(config)
    linears = 0
    for name, module in list(qwen.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, torch.nn.Linear):
                setattr(module, child_name, linear(f"model.{name}.{child_name}", child))
                linears += 1
    state = {key: weights.get_tensor("model." + key).float() for key in qwen.state_dict()}
    print(f"  {linears} matmuls, {len(state)} plain tensors")
    return qwen, state


def export_text_encoder():
    from transformers import AutoConfig

    root, shards = open_text_weights()
    weights = ShardedWeights(shards)
    config = AutoConfig.from_pretrained(root)

    def linear(full, child):
        assert child.bias is None and child.in_features % 32 == 0, full
        bits = 2 if is_ternary(weights, full + ".weight") else 4
        return NBitsLinear(full, child.out_features, child.in_features, bits)

    qwen, state = build_text_encoder(weights, config, linear)
    state["embed_tokens.weight"] = weights.get_tensor("model.embed_tokens.weight").half()
    qwen.load_state_dict(state, strict=False, assign=True)
    model = TextEncoder(qwen.eval()).eval()
    ids, mask = text_inputs()
    path = ASSETS / "text_encoder.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model,
            (ids, mask),
            str(path),
            input_names=["input_ids", "attention_mask"],
            output_names=["embeddings"],
            opset_version=18,
            custom_opsets={"com.microsoft": 1},
            dynamo=False,
        )
    written = attach_packed_weights(path, weights, (1, TEXT_LEN, 7680))
    print(f"  Exported to {path}, {written / 1e9:.2f} GB of packed weights")


def text_encoder_reference():
    """Runs the original weights in f32 on the fixed prompt for tract to compare against."""
    from transformers import AutoConfig

    root, shards = open_text_weights()
    weights = ShardedWeights(shards)
    config = AutoConfig.from_pretrained(root)
    qwen, state = build_text_encoder(
        weights, config, lambda full, child: LazyLinear(weights, full + ".weight")
    )
    qwen.load_state_dict(state, strict=False, assign=True)
    ids, mask = text_inputs()
    with torch.no_grad():
        embeddings = TextEncoder(qwen.eval())(ids, mask)
    np.savez(
        ASSETS / "text_encoder.io.npz",
        input_ids=ids.numpy(),
        attention_mask=mask.numpy(),
        embeddings=embeddings.numpy(),
    )


def main():
    component = os.environ.get("EXPORT_COMPONENT", "all")
    ASSETS.mkdir(parents=True, exist_ok=True)
    if component in ("all", "vae_decoder"):
        export_vae()
    if component in ("all", "transformer"):
        export_transformer()
    if component in ("all", "transformer_reference"):
        transformer_reference()
    if component in ("all", "text_encoder"):
        export_text_encoder()
    if component in ("all", "text_encoder_reference"):
        text_encoder_reference()


if __name__ == "__main__":
    main()
