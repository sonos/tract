"""Export NVIDIA's NemotronLabs VoiceChat (nvidia/NVIDIA-NemotronLabs-VoiceChat-11B) to NNEF for
the nemotron-voicechat example.

Writes, under <assets>/model:
- mel.nnef.tgz: 128 log-mel frames from 16 kHz audio the caller has preemphasized (0.97) and
  reflect-padded at the start of the stream: a valid stride-160 STFT, so it pulses.
- encoder.nnef.tgz: the FastConformer speech encoder with VoiceChat's weights and its
  [70, 0] attention context (no lookahead), 8 mel frames to one 80 ms frame of 1024.
- rnnt-decoder.nnef.tgz / rnnt-joint.nnef.tgz, rnnt-vocab.json: VoiceChat's RNNT head on the
  encoder, which the example runs for turn-taking and the user's transcript.
- llm.nnef.tgz: one step of the Nemotron-H 9B language model (f16) with VoiceChat's input sum
  and its text and function heads. Inputs prev_text, prev_func, prompt_id [1, S] (i64),
  prompt_mask [1, S, 1], audio [1, S, 1024], then each Mamba layer's conv / ssm state and each
  attention layer's past K / V. Outputs the greedy text and function tokens [1, S], then the
  updated states. Attention has no mask: feed one step at a time.
- tts.nnef.tgz: one step of the EAR-TTS speech generator: Gemma-3 backbone on a conditional and
  an unconditional row with an explicit KV cache, then the Mixture-of-Gaussians sampler with
  classifier-free guidance, top-p and residual quantization. Inputs code_prev [1, 1, 31],
  subword [1, 1] (i64), override [2, 1, 1152] and override_mask [1, 1, 1] (the voice warm-up),
  uniforms [1, 5, 1024] and normals [1, 5, 512] (the sampler's random draws), past K / V.
- codec.nnef.tgz: the codec decoder, 31 codes per frame to 1764 samples at 22.05 kHz, with the
  inverse STFT's overlap-add left uncropped so it pulses: drop the first 6 samples of the stream.
- meta.json (system prompt tokens, silence codes, warm-up length), warmup_f32.bin (the voice
  warm-up inputs), vocab.json (the LLM's token pieces).

Needs NVIDIA's VoiceChat code, which lives on a branch of NeMo Speech:
    pip install "nemo_toolkit[asr,nlp,tts,multimodal] @ git+https://github.com/NVIDIA-NeMo/Speech@nemotron-labs-voicechat"
then NumPy 2, torch_to_nnef and its NeMo exporter (see the README).

Usage: python export.py <assets> [--system-prompt TEXT]
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open
from torch import nn
from torch_to_nnef import export_model_to_nnef
from torch_to_nnef import torch_graph as tg
from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.op import helper
from torch_to_nnef.op.custom_extractors.base import ModuleInfoExtractor

VOICECHAT = "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B"
BASE_LLM = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
BASE_ASR = "nvidia/nemotron-speech-streaming-en-0.6b"
SYSTEM_PROMPT = (
    "You are an AI voice assistant developed by NVIDIA. "
    "Your name is NVIDIA Voice Chat. "
    "Answer in a spoken, conversational style rather than a written one. "
    "Do not repeat the same sentence over and over again. "
    "Start the conversation by greeting the user."
)
DT = torch.float16


def tract_target(dynamic_axes):
    it = TractNNEF.latest()
    it.check_io = False
    it.reify_sdpa_operator = True
    it.dynamic_axes = dynamic_axes
    return it


def checkpoint_tensors(ckpt, prefix):
    with safe_open(ckpt / "model.safetensors", "pt") as f:
        return {k[len(prefix):]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}


# ── Mel ──────────────────────────────────────────────────────────────────────────────────────

N_FFT, HOP, GUARD = 512, 160, 2.0**-24


class StreamingMel(nn.Module):
    """FilterbankFeatures as VoiceChat configures it (power spectrum, slaney filterbank, log
    with a 2^-24 guard, no normalization) as a stride-160 convolution with the windowed DFT."""

    def __init__(self, fb, window):
        super().__init__()
        left = (N_FFT - window.shape[0]) // 2
        w = F.pad(window, (left, N_FFT - window.shape[0] - left))
        n = torch.arange(N_FFT, dtype=torch.float64)
        k = torch.arange(N_FFT // 2 + 1, dtype=torch.float64)[:, None]
        angle = 2 * math.pi * k * n / N_FFT
        basis = torch.cat([torch.cos(angle), -torch.sin(angle)]) * w.double()
        self.register_buffer("basis", basis.float()[:, None, :])
        self.register_buffer("fb", fb[0])

    def forward(self, padded):
        re, im = F.conv1d(padded[:, None], self.basis, stride=HOP).chunk(2, dim=1)
        power = torch.sqrt(re * re + im * im).pow(2)
        return torch.log(torch.matmul(self.fb, power) + GUARD)


def export_mel(ckpt, out):
    pre = checkpoint_tensors(ckpt, "stt_model.perception.preprocessor.featurizer.")
    model = StreamingMel(pre["fb"], pre["window"]).eval()
    with torch.no_grad():
        export_model_to_nnef(
            model=model,
            args=(torch.randn(1, 16000),),
            file_path_export=out / "mel.nnef.tgz",
            input_names=["padded"],
            output_names=["mel"],
            inference_target=tract_target({"padded": {1: "N"}}),
            custom_extensions=["tract_assert N >= 400"],
        )


# ── Encoder and RNNT ─────────────────────────────────────────────────────────────────────────


def export_encoder_and_rnnt(ckpt, out):
    """The base streaming ASR model carrying VoiceChat's encoder and RNNT weights, through
    torch_to_nnef's NeMo exporter."""
    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict

    asr = nemo_asr.models.ASRModel.from_pretrained(BASE_ASR).eval()
    for prefix, module in [
        ("stt_model.perception.encoder.", asr.encoder),
        ("stt_model.rnnt_decoder.", asr.decoder),
        ("stt_model.rnnt_joint.", asr.joint),
    ]:
        module.load_state_dict(checkpoint_tensors(ckpt, prefix), strict=True)
    with open_dict(asr.cfg):
        asr.cfg.encoder.att_context_size = [70, 0]
    asr.encoder.set_default_att_context_size([70, 0])
    with tempfile.TemporaryDirectory() as tmp:
        nemo = Path(tmp) / "voicechat-asr.nemo"
        asr.save_to(str(nemo))
        export_dir = Path(tmp) / "export"
        subprocess.run(
            [sys.executable, "-c", "import sys; from torch_to_nnef_nemo.cli import main; sys.exit(main())",
             "-p", str(nemo), "-e", str(export_dir),
             "--skip-preprocessor", "--split-joint-decoder", "-tt", "skip"],
            check=True,
        )
        # The exporter bounds the time axis by the relative-position table's length, which
        # the streaming rewrite needs to narrow the table to the attention band.
        shutil.copy(export_dir / "encoder.nnef.tgz", out / "encoder.nnef.tgz")
        shutil.copy(export_dir / "decoder.nnef.tgz", out / "rnnt-decoder.nnef.tgz")
        shutil.copy(export_dir / "joint.nnef.tgz", out / "rnnt-joint.nnef.tgz")
    json.dump(list(asr.joint.vocabulary), open(out / "rnnt-vocab.json", "w"))


# ── Language model: Nemotron-H ───────────────────────────────────────────────────────────────


class CausalConv1dBias(nn.Module):
    """`tract_transformers_causal_conv1d_update`: depthwise causal conv1d with bias then SiLU
    over concat(state, x); x [1, C, S], state [1, C, k], weight [C, k], bias [C]. Inputs are
    told apart by shape, as torch-to-nnef replays them in its own order."""

    def forward(self, *args):
        weight = next(a for a in args if a.dim() == 2)
        bias = next(a for a in args if a.dim() == 1)
        state = next(a for a in args if a.dim() == 3 and a.shape[-1] == weight.shape[-1])
        x = next(a for a in args if a.dim() == 3 and a is not state)
        k = state.shape[-1]
        s = x.shape[-1]
        full = torch.cat([state, x], dim=-1)
        out = (full.unfold(-1, k, 1) * weight[None, :, None, :]).sum(-1)[:, :, -s:]
        return F.silu(out + bias[None, :, None]), full[:, :, -k:]


class Mamba2Recurrence(nn.Module):
    """`tract_transformers_mamba2_recurrent`: x [1, S, H, P] (already scaled by dt), b / c
    [1, S, G, N], log_decay [1, S, H], state [1, H, P, N]."""

    def forward(self, x, b, c, log_decay, state):
        x, b, c, log_decay = x.float(), b.float(), c.float(), log_decay.float()
        heads, groups = x.shape[2], b.shape[2]
        b = b.repeat_interleave(heads // groups, dim=2)
        c = c.repeat_interleave(heads // groups, dim=2)
        ys = []
        for t in range(x.shape[1]):
            decay = log_decay[:, t].exp()[:, :, None, None]
            state = state * decay + x[:, t, :, :, None] * b[:, t, :, None, :]
            ys.append((state * c[:, t, :, None, :]).sum(-1))
        return torch.stack(ys, dim=1).to(DT), state


def emit(g, name_to_tensor, op, inputs, outputs):
    nnef_inputs = tuple(helper.get_or_add_tensor_variable_in_nnef(g, i, name_to_tensor) for i in inputs)
    nnef_outputs = tuple(
        helper.add_tensor_variable_node_as_nnef_tensor(g, o, name_to_tensor, prevent_variable=True) for o in outputs
    )
    helper.cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor, graph=g, type=op, inputs=nnef_inputs, outputs=nnef_outputs,
        attribs={}, force_consistent_inputs_shapes=False,
    )
    return ["tract_transformers"]


def tensor_inputs(node):
    return [i for i in node.inputs if isinstance(i, tg.TensorVariable)]


def first_output_shaped_like(outs, shape):
    outs = list(outs)
    return outs if list(outs[0].shape) == list(shape) else outs[::-1]


class CausalConv1dBiasExtractor(ModuleInfoExtractor):
    MODULE_CLASS = CausalConv1dBias

    def convert_to_nnef(self, g, node, name_to_tensor, null_ref, torch_graph, inference_target, **kw):
        ins = tensor_inputs(node)
        if len(ins) != 4:
            raise T2NErrorNotImplemented(f"conv with bias expects 4 tensor inputs, got {len(ins)}")
        weight = next(t for t in ins if len(t.shape) == 2)
        bias = next(t for t in ins if len(t.shape) == 1)
        k = list(weight.shape)[-1]
        rank3 = [t for t in ins if len(t.shape) == 3]
        state = next(t for t in rank3 if list(t.shape)[-1] == k)
        x = next(t for t in rank3 if t is not state)
        outs = first_output_shaped_like(node.outputs, x.shape)
        return emit(g, name_to_tensor, "tract_transformers_causal_conv1d_update", (x, weight, state, bias), outs)


class Mamba2RecurrenceExtractor(ModuleInfoExtractor):
    MODULE_CLASS = Mamba2Recurrence

    def convert_to_nnef(self, g, node, name_to_tensor, null_ref, torch_graph, inference_target, **kw):
        ins = tensor_inputs(node)
        if [len(t.shape) for t in ins] != [4, 4, 4, 3, 4]:
            raise T2NErrorNotImplemented(f"unexpected mamba2 inputs {[list(t.shape) for t in ins]}")
        outs = first_output_shaped_like(node.outputs, ins[0].shape)
        return emit(g, name_to_tensor, "tract_transformers_mamba2_recurrent", ins, outs)


def param(t, dtype=DT):
    return nn.Parameter(t.to(dtype).contiguous(), requires_grad=False)


def rms_norm(x, weight, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


class Mamba2Mixer(nn.Module):
    def __init__(self, w, p, cfg):
        super().__init__()
        self.heads, self.head_dim = cfg["mamba_num_heads"], cfg["mamba_head_dim"]
        self.groups, self.d_state = cfg["n_groups"], cfg["ssm_state_size"]
        self.inner = self.heads * self.head_dim
        self.eps = cfg["layer_norm_epsilon"]
        self.in_proj = param(w[p + "in_proj.weight"])
        self.conv_w = param(w[p + "conv1d.weight"].squeeze(1))
        self.conv_b = param(w[p + "conv1d.bias"])
        self.dt_bias = param(w[p + "dt_bias"], torch.float32)
        self.a = param(-w[p + "A_log"].float().exp(), torch.float32)
        self.d = param(w[p + "D"])
        self.norm = param(w[p + "norm.weight"])
        self.out_proj = param(w[p + "out_proj.weight"])
        self.conv = CausalConv1dBias()
        self.ssm = Mamba2Recurrence()

    def forward(self, h, conv_state, ssm_state):
        s = h.shape[1]
        gn = self.groups * self.d_state
        gate, xbc, dt = F.linear(h, self.in_proj).split([self.inner, self.inner + 2 * gn, self.heads], dim=-1)
        xbc, conv_state = self.conv(conv_state, xbc.transpose(1, 2), self.conv_w, self.conv_b)
        x, b, c = xbc.transpose(1, 2).split([self.inner, gn, gn], dim=-1)
        dt = F.softplus(dt.float() + self.dt_bias)
        x = x.reshape(1, s, self.heads, self.head_dim)
        y, ssm_state = self.ssm(
            (x.float() * dt[..., None]).to(DT),
            b.reshape(1, s, self.groups, self.d_state),
            c.reshape(1, s, self.groups, self.d_state),
            dt * self.a,
            ssm_state,
        )
        y = (y + x * self.d[:, None]).reshape(1, s, self.inner)
        y = y.float() * F.silu(gate.float())
        y = y.view(1, s, self.groups, -1)
        y = (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)).view(1, s, self.inner)
        y = (y * self.norm.float()).to(DT)
        return F.linear(y, self.out_proj), conv_state, ssm_state


class Attention(nn.Module):
    def __init__(self, w, p, cfg):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.q, self.k, self.v = (param(w[p + f"{n}_proj.weight"]) for n in "qkv")
        self.o = param(w[p + "o_proj.weight"])

    def forward(self, h, past_k, past_v):
        s = h.shape[1]
        q = F.linear(h, self.q).view(1, s, self.heads, self.head_dim).transpose(1, 2)
        k = F.linear(h, self.k).view(1, s, self.kv_heads, self.head_dim).transpose(1, 2)
        v = F.linear(h, self.v).view(1, s, self.kv_heads, self.head_dim).transpose(1, 2)
        k = torch.cat([past_k, k], dim=2)
        v = torch.cat([past_v, v], dim=2)
        rep = self.heads // self.kv_heads
        att = F.scaled_dot_product_attention(q, k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1))
        return F.linear(att.transpose(1, 2).reshape(1, s, -1), self.o), k, v


class Mlp(nn.Module):
    def __init__(self, w, p):
        super().__init__()
        self.up, self.down = param(w[p + "up_proj.weight"]), param(w[p + "down_proj.weight"])

    def forward(self, h):
        return F.linear(F.relu(F.linear(h, self.up)).pow(2), self.down)


class VoiceChatLlm(nn.Module):
    """Nemotron-H with VoiceChat's input sum and heads:
        source = mask * embed(prompt_id) + (1 - mask) * proj(audio)
        x      = source + embed(prev_text) + 2 * embed(prev_func)"""

    def __init__(self, w, cfg):
        super().__init__()
        self.kinds = cfg["hybrid_override_pattern"]
        self.eps = cfg["layer_norm_epsilon"]
        self.emb = param(w["embed_tokens.weight"])
        self.norms = nn.ParameterList([param(w[f"llm.layers.{i}.norm.weight"]) for i in range(len(self.kinds))])
        mixers = []
        for i, kind in enumerate(self.kinds):
            p = f"llm.layers.{i}.mixer."
            mixers.append({"M": lambda: Mamba2Mixer(w, p, cfg), "*": lambda: Attention(w, p, cfg), "-": lambda: Mlp(w, p)}[kind]())
        self.mixers = nn.ModuleList(mixers)
        self.norm_f = param(w["llm.norm_f.weight"])
        self.lm_head = param(w["lm_head.weight"])
        self.function_head = param(w["function_head.weight"])
        self.proj_w = param(w["perception.proj.weight"])
        self.proj_b = param(w["perception.proj.bias"])

    def forward(self, prev_text, prev_func, prompt_id, prompt_mask, audio, *states):
        mask = prompt_mask.to(DT)
        source = mask * F.embedding(prompt_id, self.emb) + (1 - mask) * F.linear(audio.to(DT), self.proj_w, self.proj_b)
        h = source + F.embedding(prev_text, self.emb) + 2.0 * F.embedding(prev_func, self.emb)
        states = list(states)
        out = []
        for kind, norm, mixer in zip(self.kinds, self.norms, self.mixers):
            x = rms_norm(h, norm, self.eps)
            if kind == "-":
                h = h + mixer(x)
                continue
            y, a, b = mixer(x, states.pop(0), states.pop(0))
            h = h + y
            out += [a, b]
        h = rms_norm(h, self.norm_f, self.eps)
        return (F.linear(h, self.lm_head).argmax(-1), F.linear(h, self.function_head).argmax(-1), *out)


def llm_state_specs(cfg):
    conv_dim = cfg["mamba_num_heads"] * cfg["mamba_head_dim"] + 2 * cfg["n_groups"] * cfg["ssm_state_size"]
    specs, mamba, att = [], 0, 0
    for kind in cfg["hybrid_override_pattern"]:
        if kind == "M":
            specs.append((f"conv{mamba}", (1, conv_dim, cfg["conv_kernel"]), DT, None))
            ssm = (1, cfg["mamba_num_heads"], cfg["mamba_head_dim"], cfg["ssm_state_size"])
            specs.append((f"ssm{mamba}", ssm, torch.float32, None))
            mamba += 1
        elif kind == "*":
            for kv in "kv":
                specs.append((f"past_{kv}{att}", (1, cfg["num_key_value_heads"], 3, cfg["head_dim"]), DT, 2))
            att += 1
    return specs


def export_llm(ckpt, out):
    cfg = json.load(open(hf_hub_download(BASE_LLM, "config.json")))
    weights = {k: t.to(DT) if t.numel() > 1_000_000 else t for k, t in checkpoint_tensors(ckpt, "stt_model.").items()
               if not k.startswith(("perception.encoder", "perception.preprocessor", "rnnt_"))}
    model = VoiceChatLlm(weights, cfg).eval()
    del weights
    specs = llm_state_specs(cfg)
    names = [n for n, _, _, _ in specs]
    states = tuple(torch.randn(*shape, dtype=dt) * 0.1 for _, shape, dt, _ in specs)
    steps = ["prev_text", "prev_func", "prompt_id", "prompt_mask", "audio"]
    ids = torch.tensor([[12]])
    with torch.no_grad():
        export_model_to_nnef(
            model=model,
            args=(ids, ids, ids, torch.zeros(1, 1, 1), torch.zeros(1, 1, 1024), *states),
            file_path_export=out / "llm.nnef.tgz",
            input_names=[*steps, *names],
            output_names=["text", "func", *[f"new_{n}" for n in names]],
            inference_target=tract_target({**{n: {1: "S"} for n in steps}, **{n: {a: "P"} for n, _, _, a in specs if a is not None}}),
            custom_extensions=["tract_assert S >= 1", "tract_assert P >= 0"],
        )


# ── Speech generation: EAR-TTS ───────────────────────────────────────────────────────────────

GUIDANCE, TOP_P, NOISE, GUMBEL_EPS = 0.2, 0.95, 0.001, 1e-8


def build_tts(ckpt):
    from omegaconf import OmegaConf
    from nemo.collections.speechlm2.models.duplex_ear_tts import DuplexEARTTS

    cfg = json.load(open(ckpt / "config.json"))["model"]
    sg = OmegaConf.to_container(OmegaConf.create(cfg["speech_generation"]), resolve=True)
    tts = DuplexEARTTS(sg).eval()
    tts.load_state_dict(checkpoint_tensors(ckpt, "tts_model."), strict=True)
    return tts, cfg.get("inference_speaker_name") or "Aria"


def rvq_schedule(model, num_iter=8):
    rates = torch.linspace(0.0, 1.0, num_iter + 1)[:-1]
    masks = torch.ceil((1 - rates.pow(model.config.exponent)).pow(1 / model.config.exponent) * model.config.num_quantizers).long()
    ks = (masks - F.pad(masks[1:], [0, 1])).tolist()
    return [k for k in ks if k > 0]


def cas_table(model, vocab_size, batch=8192):
    """The character-aware subword embedding of every LLM token, precomputed."""
    rows = []
    with torch.no_grad():
        for start in range(0, vocab_size, batch):
            ids = torch.arange(start, min(start + batch, vocab_size))[:, None]
            rows.append(model.embed_subword(ids, torch.ones_like(ids, dtype=torch.bool))[:, 0])
    return torch.cat(rows)


def gemma_rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w)


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class GemmaStep(nn.Module):
    """HF Gemma3TextModel without its embeddings, one position, K / V in and out."""

    def __init__(self, backbone):
        super().__init__()
        c = backbone.config
        self.layers = backbone.layers
        self.norm_w = backbone.norm.weight
        self.eps = c.rms_norm_eps
        self.heads, self.kv_heads, self.head_dim = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        self.scale = c.query_pre_attn_scalar**-0.5
        self.layer_types = list(c.layer_types)
        for name, theta in [("full", c.rope_theta), ("local", c.rope_local_base_freq)]:
            inv = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            self.register_buffer(f"inv_{name}", inv)

    def forward(self, x, kv):
        b, s, _ = x.shape
        p = kv[0].shape[2]
        pos = (torch.arange(s) + p).float()[:, None]
        rope = {}
        for name in ("full", "local"):
            f = pos * getattr(self, f"inv_{name}")[None, :]
            e = torch.cat((f, f), dim=-1)
            rope[name] = (e.cos(), e.sin())
        out = []
        for i, layer in enumerate(self.layers):
            a = layer.self_attn
            cos, sin = rope["full" if self.layer_types[i] == "full_attention" else "local"]
            h = gemma_rms(x, layer.input_layernorm.weight, self.eps)
            q = a.q_proj(h).view(b, s, self.heads, self.head_dim).transpose(1, 2)
            k = a.k_proj(h).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
            v = a.v_proj(h).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
            q = gemma_rms(q, a.q_norm.weight, self.eps)
            k = gemma_rms(k, a.k_norm.weight, self.eps)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            k = torch.cat([kv[2 * i], k], dim=2)
            v = torch.cat([kv[2 * i + 1], v], dim=2)
            out += [k, v]
            att = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
            att = a.o_proj(att.transpose(1, 2).reshape(b, s, -1))
            x = x + gemma_rms(att, layer.post_attention_layernorm.weight, self.eps)
            h = gemma_rms(x, layer.pre_feedforward_layernorm.weight, self.eps)
            x = x + gemma_rms(layer.mlp(h), layer.post_feedforward_layernorm.weight, self.eps)
        return gemma_rms(x, self.norm_w, self.eps), out


class TtsStep(nn.Module):
    def __init__(self, tts, vocab_size):
        super().__init__()
        m = tts.tts_model
        self.embed_code = m.embed_code
        self.null_emb = m.null_emb
        self.fusion = m.gated_fusion_audio_text
        self.head = m.mog_head
        self.backbone = GemmaStep(m.backbone)
        self.register_buffer("cas", cas_table(m, vocab_size))
        self.register_buffer("rvq_pad", F.pad(m.rvq_embs, [0, 0, 0, 1]))
        self.register_buffer("rvq_sq", m.rvq_embs.pow(2).sum(-1))
        head = m.mog_head
        n, d = head.num_predictions, head.low_rank
        self.register_buffer("mus", head.proj_mus.weight.detach().view(n, d, -1))
        self.register_buffer("low", head.low_mat.detach().view(n, head.out_size, -1))
        self.min_log_std = head.min_log_std
        self.ks = rvq_schedule(m)
        self.nq = m.config.num_quantizers

    def fuse(self, code_prev, subword):
        a = sum(F.embedding(code_prev[..., q], self.rvq_pad[q]) for q in range(self.nq))
        a = self.embed_code(a)
        cond = F.embedding(subword, self.cas)
        uncond = cond * 0.0 + self.null_emb  # expand_as would export the traced length
        return self.fusion(torch.cat([a, a]), torch.cat([cond, uncond]))

    def sample(self, h, uniforms, normals):
        head = self.head
        h_c, h_u = h[:1], h[1:]
        filled = torch.zeros_like(h_c[..., :1]).expand(1, 1, self.rvq_pad.shape[-1]) * 0.0
        codes = []
        q = 0
        for it, k in enumerate(self.ks):
            e = self.embed_code(filled)
            x = head.mlp_stack(torch.cat([e + h_c, e + h_u]))
            x = x[:1] + GUIDANCE * (x[:1] - x[1:])
            logits = head.proj_logits(x)[0]
            p = logits.softmax(-1)
            below = (logits <= logits.transpose(0, 1)).float()
            removed = (below @ p.transpose(0, 1)).transpose(0, 1) <= (1 - TOP_P)
            logits = logits.masked_fill(removed, float("-inf"))
            gumbel = -torch.log(-torch.log(uniforms[0, it : it + 1] + GUMBEL_EPS) + GUMBEL_EPS)
            # log_softmax only shifts the logits by a constant: the argmax is the same without it.
            j = (logits + gumbel).argmax(-1)
            xv = x[0]
            mu = xv @ self.mus.index_select(0, j)[0].transpose(0, 1)
            mu = mu @ self.low.index_select(0, j)[0].transpose(0, 1)
            logs = head.proj_logs(xv).clamp_min(self.min_log_std)
            z = mu * torch.exp(logs) + head.proj_else(xv) + torch.exp(logs) * normals[0, it : it + 1] * NOISE
            for _ in range(k):
                table = self.rvq_pad[q, :-1]
                c = (self.rvq_sq[q : q + 1] - 2 * (z @ table.transpose(0, 1))).argmin(-1)
                e_q = table.index_select(0, c)
                z = z - e_q
                filled = filled + e_q[None]
                codes.append(c)
                q += 1
        return torch.cat(codes)[None]

    def forward(self, code_prev, subword, override, override_mask, uniforms, normals, *kv):
        u = override_mask * override + (1 - override_mask) * self.fuse(code_prev, subword)
        h, kv = self.backbone(u, kv)
        return (self.sample(h[:, -1:], uniforms, normals), *kv)


def voice_warmup(tts, speaker):
    """The backbone inputs of the speaker prompt NVIDIA's TTS runs before the first step."""
    captured = []
    hook = tts.tts_model.backbone.register_forward_hook(
        lambda module, args, kwargs, output: captured.append(kwargs["inputs_embeds"].detach().clone()), with_kwargs=True
    )
    with torch.no_grad():
        tts.set_init_inputs(speaker_name=speaker)
        init = tts.get_init_inputs(B=1)
        init.update({"use_cache": True, "past_key_values": None, "guidance_enabled": True})
        tts.tts_model(**init)
    hook.remove()
    return captured[0]


def export_tts(ckpt, out, vocab_size):
    tts, speaker = build_tts(ckpt)
    warm = voice_warmup(tts, speaker)
    model = TtsStep(tts, vocab_size).eval()
    layers, heads, dim = len(model.backbone.layers), model.backbone.kv_heads, model.backbone.head_dim
    kv_names = [f"past_{x}{i}" for i in range(layers) for x in "kv"]
    kv = tuple(torch.randn(2, heads, 3, dim) for _ in range(2 * layers))
    args = (
        torch.zeros(1, 1, 31, dtype=torch.long),
        torch.zeros(1, 1, dtype=torch.long),
        torch.zeros(2, 1, 1152),
        torch.zeros(1, 1, 1),
        torch.rand(1, 5, 1024),
        torch.randn(1, 5, 512),
        *kv,
    )
    steps = ["code_prev", "subword", "override", "override_mask"]
    with torch.no_grad():
        export_model_to_nnef(
            model=model,
            args=args,
            file_path_export=out / "tts.nnef.tgz",
            input_names=[*steps, "uniforms", "normals", *kv_names],
            output_names=["codes", *[f"new_{n}" for n in kv_names]],
            inference_target=tract_target({**{n: {2: "P"} for n in kv_names}, **{n: {1: "S"} for n in steps}}),
            custom_extensions=["tract_assert P >= 0", "tract_assert S >= 1"],
        )
    return warm.float().numpy(), tts.codec_silence_tokens.tolist()


# ── Codec decoder ────────────────────────────────────────────────────────────────────────────

CODEC_FFT, CODEC_HOP = 16, 4


def irfft_basis(n):
    """x[t] = A @ re + B @ im for the n-point inverse real FFT ("backward" norm)."""
    k = np.arange(n // 2 + 1)[None, :]
    t = np.arange(n)[:, None]
    scale = np.full(n // 2 + 1, 2.0)
    scale[0] = scale[-1] = 1.0
    a = scale * np.cos(2 * np.pi * k * t / n) / n
    b = -scale * np.sin(2 * np.pi * k * t / n) / n
    b[:, 0] = b[:, -1] = 0.0
    return torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)


class CodecDecoder(nn.Module):
    """NVIDIA's decoder layers, with the dequantization and the inverse STFT rewritten: a lookup
    per codebook, a fixed inverse-DFT matrix, overlap-add and its window envelope as stride-hop
    transposed convolutions. The overlap-add is not cropped, so the graph pulses."""

    def __init__(self, codec):
        super().__init__()
        self.mus = nn.Parameter(torch.stack(list(codec.prvq.mus_list)), requires_grad=False)
        self.layers = codec.decoder.layers
        a, b = irfft_basis(CODEC_FFT)
        self.register_buffer("a", a)
        self.register_buffer("b", b)
        window = torch.hann_window(CODEC_FFT)
        self.register_buffer("window", window[None, :, None])
        self.register_buffer("window_sq", window.square()[None, :, None])
        self.register_buffer("ola", torch.eye(CODEC_FFT)[:, None, :])

    def forward(self, codes):
        z = sum(F.embedding(codes[:, :, q], self.mus[q]) for q in range(self.mus.shape[0]))
        x = z.transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        mag, ph = x.chunk(2, dim=1)
        mag = 100.0 * torch.exp(-F.softplus(-mag + math.log(100.0)))
        frames = torch.einsum("nk,bkt->bnt", self.a, mag * torch.cos(ph)) + torch.einsum(
            "nk,bkt->bnt", self.b, mag * torch.sin(ph)
        )
        frames = torch.maximum(torch.minimum(frames, self.window), -self.window) * self.window
        wav = F.conv_transpose1d(frames, self.ola, stride=CODEC_HOP)
        # frames * 0 broadcasts the squared window to the symbolic length (expand_as would
        # export the traced one).
        envelope = F.conv_transpose1d(frames * 0.0 + self.window_sq, self.ola, stride=CODEC_HOP)
        return wav / envelope


def export_codec(ckpt, out):
    from omegaconf import DictConfig
    from nemo.collections.speechlm2.modules.ear_tts_vae_codec import RVQVAEModel

    cfg = json.load(open(ckpt / "config.json"))["model"]["speech_generation"]["model"]["codec_config"]
    codec = RVQVAEModel(DictConfig(cfg)).eval()
    codec.load_state_dict(checkpoint_tensors(ckpt, "tts_model.audio_codec."), strict=False)
    with torch.no_grad():
        export_model_to_nnef(
            model=CodecDecoder(codec).eval(),
            args=(torch.zeros(1, 4, 31, dtype=torch.long),),
            file_path_export=out / "codec.nnef.tgz",
            input_names=["codes"],
            output_names=["audio"],
            inference_target=tract_target({"codes": {1: "T"}}),
            custom_extensions=["tract_assert T >= 1"],
        )


# ── Tokens ───────────────────────────────────────────────────────────────────────────────────


def export_tokens(out, system_prompt):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_LLM)
    vocab_size = json.load(open(hf_hub_download(BASE_LLM, "config.json")))["vocab_size"]
    vocab = [tok.convert_tokens_to_string([tok.convert_ids_to_tokens(i)]) for i in range(vocab_size)]
    json.dump(vocab, open(out / "vocab.json", "w"))
    # VoiceChat delimits the prompt with <s> and </s>, not the tokenizer's own end token.
    bos, eos = tok.convert_tokens_to_ids(["<s>", "</s>"])
    prompt = [bos, *tok.encode(system_prompt, add_special_tokens=False), eos]
    return vocab_size, prompt


def main(args):
    out = Path(args.assets) / "model"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = Path(snapshot_download(VOICECHAT))
    vocab_size, prompt = export_tokens(out, args.system_prompt)
    stages = [
        ("mel.nnef.tgz", lambda: export_mel(ckpt, out)),
        ("rnnt-joint.nnef.tgz", lambda: export_encoder_and_rnnt(ckpt, out)),
        ("codec.nnef.tgz", lambda: export_codec(ckpt, out)),
        ("llm.nnef.tgz", lambda: export_llm(ckpt, out)),
    ]
    for done, stage in stages:
        if not (out / done).exists():
            print(f"exporting {done}", flush=True)
            stage()
    meta = out / "meta.json"
    if not meta.exists():
        print("exporting tts.nnef.tgz", flush=True)
        warm, silence = export_tts(ckpt, out, vocab_size)
        warm.astype("<f4").tofile(out / "warmup_f32.bin")
        json.dump({"prompt": prompt, "silence": silence, "warm_len": int(warm.shape[1])}, open(meta, "w"))
    else:
        m = json.load(open(meta))
        m["prompt"] = prompt
        json.dump(m, open(meta, "w"))
    print(f"done: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("assets")
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    args = parser.parse_args()
    # The language model and the TTS trace deep graphs.
    sys.setrecursionlimit(1_000_000)
    threading.stack_size(1 << 30)
    failure = []

    def run():
        try:
            main(args)
        except BaseException as e:
            failure.append(e)
            raise

    worker = threading.Thread(target=run)
    worker.start()
    worker.join()
    sys.exit(1 if failure else 0)
