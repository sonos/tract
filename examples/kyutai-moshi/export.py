"""Export Kyutai's Moshi (kyutai/moshiko-pytorch-bf16) to NNEF for the kyutai-moshi example.

Writes, under <assets>/model:
- mimi-encoder / mimi-decoder: the Mimi codec, as in the kyutai-mimi example
- helium.nnef.tgz: the 7B main transformer, f16, a step graph with a KV cache. Inputs:
  tokens [1, 17, S] (text + 16 audio streams), past_k{i} / past_v{i} [1, 32, P, 128].
  Outputs: hidden [1, S, 4096] (after out_norm), text_logits [1, S, 32000], k{i} / v{i}
  [1, 32, P + S, 128], which tract folds into a stateful KV cache (it needs S symbolic).
  Feed it one frame per step (S = 1): a single query may attend to the whole past, so the
  graph has no mask, which holds up to Helium's 3000-frame context.
- depformer-step{0..7}.nnef.tgz: the depth transformer, f16, one graph per step, each with
  that step's weights. Inputs: token [1, 1], hidden [1, 1, 4096], past K/V of the earlier
  steps. Outputs: logits [1, 2048] and the K/V extended by one position.
- vocab.json: the text tokenizer's pieces.

With --check, also runs Kyutai's own LMGen greedily over the first frames of the sample and
writes <assets>/reference/moshi_greedy.json for the example's --check. This needs Kyutai's
`moshi` package and about 60 GB of RAM (the model runs in f32 on the CPU).

Usage: python export.py <assets> <sample.wav> [--check]
"""

import importlib.util
import json
import math
import sys
import threading
from pathlib import Path

import sentencepiece
import soundfile as sf
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch_to_nnef import export_model_to_nnef
from transformers import MimiModel

REPO = "kyutai/moshiko-pytorch-bf16"
DIM, HEADS, LAYERS, HIDDEN = 4096, 32, 32, 11264
HEAD_DIM = DIM // HEADS
DEP_DIM, DEP_HEADS, DEP_LAYERS, DEP_HIDDEN, DEP_STEPS = 1024, 16, 6, 2816, 8
DEP_HEAD_DIM = DEP_DIM // DEP_HEADS
CHECK_FRAMES = 20
DT = torch.float16

spec = importlib.util.spec_from_file_location("mimi_export", Path(__file__).parent.parent / "kyutai-mimi" / "export.py")
mimi_export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mimi_export)


def param(t):
    return torch.nn.Parameter(t.to(DT).contiguous(), requires_grad=False)


def rms_norm(x, alpha):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-8) * alpha.float()
    return y.to(x.dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def embed(table, tokens):
    return F.embedding(tokens, table)


class Helium(torch.nn.Module):
    """Moshi's main transformer. Moshi's RoPE rotates interleaved (even, odd) pairs; the q and
    k rows are permuted per head to the half-split layout, which leaves the attention scores
    unchanged and gives the rotate-half form tract fuses."""

    def __init__(self, weights):
        super().__init__()
        self.text_emb = param(weights["text_emb.weight"])
        self.emb = torch.nn.ParameterList([param(weights[f"emb.{i}.weight"]) for i in range(16)])
        perm = torch.cat([torch.arange(0, HEAD_DIM, 2), torch.arange(1, HEAD_DIM, 2)])
        perm = (torch.arange(HEADS)[:, None] * HEAD_DIM + perm[None]).flatten()
        self.layers = torch.nn.ModuleList()
        for i in range(LAYERS):
            p = f"transformer.layers.{i}."
            q, k, v = weights[p + "self_attn.in_proj_weight"].split(DIM)
            layer = torch.nn.Module()
            layer.norm1 = param(weights[p + "norm1.alpha"])
            layer.norm2 = param(weights[p + "norm2.alpha"])
            layer.q, layer.k, layer.v = param(q[perm]), param(k[perm]), param(v)
            layer.o = param(weights[p + "self_attn.out_proj.weight"])
            layer.gate_in = param(weights[p + "gating.linear_in.weight"])
            layer.gate_out = param(weights[p + "gating.linear_out.weight"])
            self.layers.append(layer)
        self.out_norm = param(weights["out_norm.alpha"])
        self.text_linear = param(weights["text_linear.weight"])
        inv_freq = torch.exp(torch.arange(HEAD_DIM // 2) * (-math.log(10000) * 2 / HEAD_DIM))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tokens, *past):
        s = tokens.shape[2]
        p = past[0].shape[2]
        x = embed(self.text_emb, tokens[:, 0])
        for i in range(16):
            x = x + embed(self.emb[i], tokens[:, i + 1])
        freqs = (torch.arange(s) + p).float()[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(DT), emb.sin().to(DT)
        present = []
        for i, layer in enumerate(self.layers):
            h = rms_norm(x, layer.norm1)
            q = F.linear(h, layer.q).view(1, s, HEADS, HEAD_DIM).transpose(1, 2)
            k = F.linear(h, layer.k).view(1, s, HEADS, HEAD_DIM).transpose(1, 2)
            v = F.linear(h, layer.v).view(1, s, HEADS, HEAD_DIM).transpose(1, 2)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            k = torch.cat([past[2 * i], k], dim=2)
            v = torch.cat([past[2 * i + 1], v], dim=2)
            present += [k, v]
            att = F.scaled_dot_product_attention(q, k, v)
            x = x + F.linear(att.transpose(1, 2).reshape(1, s, DIM), layer.o)
            g = F.linear(rms_norm(x, layer.norm2), layer.gate_in).view(1, s, 2, HIDDEN)
            x = x + F.linear(F.silu(g[:, :, 0]) * g[:, :, 1], layer.gate_out)
        hidden = rms_norm(x, self.out_norm)
        return (hidden, F.linear(hidden, self.text_linear), *present)


class DepformerStep(torch.nn.Module):
    """Step k of the depth transformer, with its own slice of every per-step weight."""

    def __init__(self, weights, k):
        super().__init__()
        step = lambda name: weights[name].view(DEP_STEPS, -1, *weights[name].shape[1:])[k]
        self.depformer_in = param(weights[f"depformer_in.{k}.weight"])
        self.emb = param(weights["depformer_text_emb.weight" if k == 0 else f"depformer_emb.{k - 1}.weight"])
        self.layers = torch.nn.ModuleList()
        for i in range(DEP_LAYERS):
            p = f"depformer.layers.{i}."
            layer = torch.nn.Module()
            layer.norm1 = param(weights[p + "norm1.alpha"])
            layer.norm2 = param(weights[p + "norm2.alpha"])
            layer.in_proj = param(step(p + "self_attn.in_proj_weight"))
            layer.out_proj = param(step(p + "self_attn.out_proj.weight"))
            layer.gate_in = param(weights[p + f"gating.{k}.linear_in.weight"])
            layer.gate_out = param(weights[p + f"gating.{k}.linear_out.weight"])
            self.layers.append(layer)
        self.head = param(weights[f"linears.{k}.weight"])

    def forward(self, token, hidden, *past):
        x = F.linear(hidden, self.depformer_in) + embed(self.emb, token)
        kv = []
        for i, layer in enumerate(self.layers):
            h = rms_norm(x, layer.norm1)
            q, k, v = F.linear(h, layer.in_proj).view(1, 1, 3, DEP_HEADS, DEP_HEAD_DIM).unbind(2)
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))
            if past:
                k = torch.cat([past[2 * i], k], dim=2)
                v = torch.cat([past[2 * i + 1], v], dim=2)
            att = ((q @ k.transpose(2, 3)) / DEP_HEAD_DIM**0.5).softmax(-1) @ v
            x = x + F.linear(att.transpose(1, 2).reshape(1, 1, DEP_DIM), layer.out_proj)
            g = F.linear(rms_norm(x, layer.norm2), layer.gate_in).view(1, 1, 2, DEP_HIDDEN)
            x = x + F.linear(F.silu(g[:, :, 0]) * g[:, :, 1], layer.gate_out)
            kv += [k, v]
        return (F.linear(x, self.head)[:, 0], *kv)


def target():
    it = mimi_export.target()
    it.check_io = False
    return it


def load(path, keep):
    with safe_open(path, "pt") as f:
        return {k: f.get_tensor(k) for k in f.keys() if keep(k)}


def export_mimi(model_dir, audio):
    mimi = MimiModel.from_pretrained("kyutai/mimi").eval()
    encoder = mimi_export.Encoder(mimi, mimi_export.CODEBOOKS).eval()
    decoder = mimi_export.Decoder(mimi, mimi_export.CODEBOOKS).eval()
    with torch.no_grad():
        codes = encoder(audio)
    for name, model, arg, inp, out, axis, sym in [
        ("encoder", encoder, audio, "audio", "codes", 2, "T"),
        ("decoder", decoder, codes, "codes", "audio", 2, "F"),
    ]:
        it = mimi_export.target()
        it.dynamic_axes = {inp: {axis: sym}}
        minimum = mimi_export.FRAME if sym == "T" else 1
        export_model_to_nnef(
            model=model,
            args=(arg,),
            file_path_export=model_dir / f"mimi-{name}.nnef.tgz",
            input_names=[inp],
            output_names=[out],
            inference_target=it,
            custom_extensions=[f"tract_assert {sym} >= {minimum}"],
        )
    return codes


def export_helium(weights_path, model_dir):
    model = Helium(load(weights_path, lambda k: not k.startswith(("depformer", "linears")))).eval()
    past = tuple(torch.randn(1, HEADS, 3, HEAD_DIM, dtype=DT) for _ in range(2 * LAYERS))
    kv_in = [f"past_{kv}{i}" for i in range(LAYERS) for kv in "kv"]
    it = target()
    it.reify_sdpa_operator = True
    it.dynamic_axes = {"tokens": {2: "S"}, **{n: {2: "P"} for n in kv_in}}
    with torch.no_grad():
        export_model_to_nnef(
            model=model,
            args=(torch.randint(0, 2048, (1, 17, 1)), *past),
            file_path_export=model_dir / "helium.nnef.tgz",
            input_names=["tokens", *kv_in],
            output_names=["hidden", "text_logits", *[f"{kv}{i}" for i in range(LAYERS) for kv in "kv"]],
            inference_target=it,
            custom_extensions=["tract_assert S >= 1", "tract_assert P >= 0"],
        )


def export_depformer(weights_path, model_dir):
    weights = load(weights_path, lambda k: k.startswith(("depformer", "linears")))
    for k in range(DEP_STEPS):
        past = tuple(torch.randn(1, DEP_HEADS, k, DEP_HEAD_DIM, dtype=DT) for _ in range(2 * DEP_LAYERS))
        kv_in = [f"past_{kv}{i}" for i in range(DEP_LAYERS) for kv in "kv"] if k else []
        with torch.no_grad():
            export_model_to_nnef(
                model=DepformerStep(weights, k).eval(),
                args=(torch.tensor([[3]]), torch.randn(1, 1, DIM, dtype=DT), *(past if k else ())),
                file_path_export=model_dir / f"depformer-step{k}.nnef.tgz",
                input_names=["token", "hidden", *kv_in],
                output_names=["logits", *[f"{kv}{i}" for i in range(DEP_LAYERS) for kv in "kv"]],
                inference_target=target(),
            )


def greedy_reference(weights_path, codes, path):
    from moshi.models import loaders
    from moshi.models.lm import LMGen

    lm = loaders.get_moshi_lm(weights_path, device="cpu", dtype=torch.float32).eval()
    gen = LMGen(lm, use_sampling=False)
    codes = codes[:, :, :CHECK_FRAMES]
    outputs = []
    with torch.no_grad(), gen.streaming(1):
        for t in range(codes.shape[2]):
            out = gen.step(codes[:, :, t : t + 1])
            outputs.append(None if out is None else out[0, :, 0].tolist())
    path.write_text(json.dumps({"user_codes": codes[0].tolist(), "outputs": outputs}))


def main(assets, wav, check):
    model_dir = assets / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    weights_path = hf_hub_download(REPO, "model.safetensors")
    spm = sentencepiece.SentencePieceProcessor(model_file=hf_hub_download(REPO, "tokenizer_spm_32k_3.model"))
    (model_dir / "vocab.json").write_text(json.dumps([spm.id_to_piece(i) for i in range(spm.get_piece_size())]))

    signal, sr = sf.read(wav, dtype="float32")
    signal = torch.from_numpy(signal)[None, None]
    if sr != 24000:
        signal = F.interpolate(signal, scale_factor=24000 / sr, mode="linear", align_corners=False)
    audio = signal[..., : signal.shape[-1] // mimi_export.FRAME * mimi_export.FRAME].contiguous()

    codes = export_mimi(model_dir, audio)
    export_helium(weights_path, model_dir)
    export_depformer(weights_path, model_dir)
    if check:
        (assets / "reference").mkdir(exist_ok=True)
        greedy_reference(weights_path, codes, assets / "reference" / "moshi_greedy.json")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--check"]
    # torch-to-nnef walks the traced graph recursively, deeper than Python's default limit
    # on 32 layers: give it room, on a thread with a stack to match.
    sys.setrecursionlimit(1_000_000)
    threading.stack_size(1 << 30)
    failure = []

    def run():
        try:
            main(Path(args[0]), Path(args[1]), "--check" in sys.argv)
        except BaseException as e:
            failure.append(e)
            raise

    worker = threading.Thread(target=run)
    worker.start()
    worker.join()
    sys.exit(1 if failure else 0)
