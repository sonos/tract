"""Export mistralai/Voxtral-Mini-4B-Realtime-2602 to NNEF for the voxtral-realtime example.

Writes, under <assets>/model:
- encoder.nnef.tgz: 16 kHz audio [1, 1, T] -> audio embeddings [1, T / 1280, 3072]. Log-mel,
  causal convolutions, causal attention over a 750-frame sliding window, and the projector
  (its 4-frame concatenation as a stride-4 convolution): tract pulses it to one 80 ms chunk.
- decoder.nnef.tgz: one step of the language model. token [1, S] i64, audio [1, S, 3072],
  past_k{i} / past_v{i} [1, 8, P, 128] -> logits [1, 131072], k{i} / v{i} [1, 8, P + S, 128],
  which tract folds into a stateful KV cache. Feed it one position per step (S = 1): a single
  query may attend to the whole past, so the graph has no mask. The time conditioning of its
  norms is folded in for the chosen transcription delay.
- vocab.json: the bytes of every text token, hex-encoded, and config.json: the delay.

Without a sample, downloads a LibriSpeech one to <assets>/2086-149220-0033.wav, which the
example transcribes by default. With --check, also runs the transformers implementation greedily over the sample and writes
the tokens it generates to <assets>/reference/tokens.json, for the example's --check. This
runs the model in f32 on the CPU: plan for about 20 GB of RAM.

Usage: python export.py <assets> [sample.wav] [--delay-ms 480] [--check]
"""

import argparse
import base64
import json
import math
import os
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from torch_to_nnef import export_model_to_nnef
from torch_to_nnef.inference_target import TractNNEF
from transformers import AutoProcessor
from transformers import VoxtralRealtimeForConditionalGeneration

REPO = "mistralai/Voxtral-Mini-4B-Realtime-2602"
SAMPLE_URL = "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav"
LOOK_AHEAD = 40


def target():
    tract = os.environ.get("TRACT_BIN")
    it = TractNNEF.latest() if tract is None else TractNNEF(
        subprocess.check_output([tract, "--version"], text=True).split()[-1],
        specific_tract_binary_path=Path(tract),
    )
    it.check_io = False
    return it


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rms_norm(x, w, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def inv_freq(head_dim, theta):
    return 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))


def causal_conv(x, conv):
    k = (conv.kernel_size[0] - 1) * conv.dilation[0] + 1
    x = F.pad(x, (k - conv.stride[0], 0))
    return F.conv1d(x, conv.weight, conv.bias, conv.stride, 0, conv.dilation)


class Mel(torch.nn.Module):
    """The log-mel of VoxtralRealtimeFeatureExtractor. Its frame i is the 400-sample Hann
    window centred on sample 160 i, so it reads 40 samples past its hop: the graph takes its
    input LOOK_AHEAD samples early, which keeps it causal with one frame per 160 samples."""

    def __init__(self, mel_filters):
        super().__init__()
        n = torch.arange(400, dtype=torch.float64)
        f = torch.arange(201, dtype=torch.float64)
        ang = 2 * math.pi * f[:, None] * n[None, :] / 400
        basis = torch.cat([torch.cos(ang), -torch.sin(ang)]) * torch.hann_window(400, dtype=torch.float64)
        self.register_buffer("basis", basis[:, None, :].float())
        self.register_buffer("filters", torch.from_numpy(mel_filters).float().T.contiguous())

    def forward(self, audio):
        spec = F.conv1d(F.pad(audio, (200 + LOOK_AHEAD, 0)), self.basis, stride=160)
        power = spec[:, :201] ** 2 + spec[:, 201:] ** 2
        log = torch.log10(torch.clamp(torch.matmul(self.filters, power), min=1e-10))
        log = torch.maximum(log, torch.full_like(log, 1.5 - 8.0))
        return (log + 4.0) / 4.0


class Encoder(torch.nn.Module):
    def __init__(self, model, mel_filters):
        super().__init__()
        cfg = model.config.audio_config
        self.mel = Mel(mel_filters)
        self.tower = model.model.audio_tower
        self.heads, self.head_dim, self.window = cfg.num_attention_heads, cfg.head_dim, cfg.sliding_window
        self.register_buffer("inv_freq", inv_freq(cfg.head_dim, cfg.rope_parameters["rope_theta"]))
        proj = model.model.multi_modal_projector
        d, f = cfg.hidden_size, model.config.downsample_factor
        # linear_1 reads 4 frames concatenated: input index frame * d + channel
        w1 = proj.linear_1.weight.data.view(-1, f, d).permute(0, 2, 1).contiguous()
        self.proj1 = torch.nn.Parameter(w1, requires_grad=False)
        self.proj2 = proj.linear_2
        self.factor = f

    def forward(self, audio):
        emb = self.tower.embedder
        x = F.gelu(causal_conv(self.mel(audio), emb.conv1))
        x = F.gelu(causal_conv(x, emb.conv2)).transpose(1, 2)
        t = x.shape[1]
        freqs = torch.arange(t).to(torch.float32)[:, None] * self.inv_freq[None, :]
        e = torch.cat((freqs, freqs), dim=-1)
        cos, sin = e.cos(), e.sin()
        idx = torch.arange(t)
        diff = idx[:, None] - idx[None, :]
        mask = (diff >= 0) & (diff < self.window)
        for layer in self.tower.layers:
            att = layer.self_attn
            h = rms_norm(x, layer.self_attn_layer_norm.weight)
            q, k, v = (p(h).view(1, t, self.heads, self.head_dim).transpose(1, 2) for p in (att.q_proj, att.k_proj, att.v_proj))
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            scores = (q @ k.transpose(2, 3)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~mask, float("-inf"))
            o = (scores.softmax(-1) @ v).transpose(1, 2).reshape(1, t, self.heads * self.head_dim)
            x = x + att.o_proj(o)
            h = rms_norm(x, layer.final_layer_norm.weight)
            m = layer.mlp
            x = x + m.down_proj(F.silu(m.gate_proj(h)) * m.up_proj(h))
        x = rms_norm(x, self.tower.norm.weight)
        y = F.conv1d(x.transpose(1, 2), self.proj1, stride=self.factor).transpose(1, 2)
        return self.proj2(F.gelu(y))


class Decoder(torch.nn.Module):
    def __init__(self, model, delay_tokens):
        super().__init__()
        cfg = model.config.text_config
        lm = model.model.language_model
        self.heads, self.kv_heads, self.head_dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.embed = lm.embed_tokens.weight
        self.layers = lm.layers
        self.norm = lm.norm.weight
        t_cond = model.model.time_embedding(torch.full((1,), float(delay_tokens)))
        self.mlp_norms = torch.nn.ParameterList(
            [
                torch.nn.Parameter(l.post_attention_layernorm.weight * (1 + l.ada_rms_norm(t_cond)), requires_grad=False)
                for l in self.layers
            ]
        )
        self.register_buffer("inv_freq", inv_freq(cfg.head_dim, cfg.rope_parameters["rope_theta"]))

    def forward(self, token, audio, *past):
        s, p = token.shape[1], past[0].shape[2]
        kv, group, hd = self.kv_heads, self.heads // self.kv_heads, self.head_dim
        x = F.embedding(token, self.embed) + audio
        freqs = (torch.arange(s) + p).float()[:, None] * self.inv_freq[None, :]
        e = torch.cat((freqs, freqs), dim=-1)
        cos, sin = e.cos(), e.sin()
        present = []
        for i, layer in enumerate(self.layers):
            att = layer.self_attn
            h = rms_norm(x, layer.input_layernorm.weight)
            q = att.q_proj(h).view(1, s, self.heads, hd).transpose(1, 2)
            k = att.k_proj(h).view(1, s, kv, hd).transpose(1, 2)
            v = att.v_proj(h).view(1, s, kv, hd).transpose(1, 2)
            q = q * cos + rotate_half(q) * sin
            k = torch.cat([past[2 * i], k * cos + rotate_half(k) * sin], dim=2)
            v = torch.cat([past[2 * i + 1], v], dim=2)
            present += [k, v]
            # one query position: the query heads sharing a kv head attend as `group` positions
            o = F.scaled_dot_product_attention(q.reshape(1, kv, group * s, hd), k, v)
            x = x + att.o_proj(o.reshape(1, self.heads, s, hd).transpose(1, 2).reshape(1, s, self.heads * hd))
            h = rms_norm(x, self.mlp_norms[i])
            m = layer.mlp
            x = x + m.down_proj(F.silu(m.gate_proj(h)) * m.up_proj(h))
        x = rms_norm(x[:, -1], self.norm)
        return (F.linear(x, self.embed), *present)


def export_encoder(model, processor, model_dir):
    it = target()
    it.dynamic_axes = {"audio": {2: "T"}}
    export_model_to_nnef(
        model=Encoder(model, processor.feature_extractor.mel_filters).eval(),
        args=(torch.randn(1, 1, 64 * 160),),
        file_path_export=model_dir / "encoder.nnef.tgz",
        input_names=["audio"],
        output_names=["embeddings"],
        inference_target=it,
        custom_extensions=["tract_assert T >= 1280"],
    )


def export_decoder(model, delay_tokens, model_dir):
    layers = model.config.text_config.num_hidden_layers
    past = tuple(torch.randn(1, 8, 3, 128) for _ in range(2 * layers))
    kv_in = [f"past_{kv}{i}" for i in range(layers) for kv in "kv"]
    it = target()
    it.reify_sdpa_operator = True
    it.dynamic_axes = {"token": {1: "S"}, "audio": {1: "S"}, **{n: {2: "P"} for n in kv_in}}
    export_model_to_nnef(
        model=Decoder(model, delay_tokens).eval(),
        args=(torch.tensor([[32]]), torch.randn(1, 1, 3072), *past),
        file_path_export=model_dir / "decoder.nnef.tgz",
        input_names=["token", "audio", *kv_in],
        output_names=["logits", *[f"{kv}{i}" for i in range(layers) for kv in "kv"]],
        inference_target=it,
        custom_extensions=["tract_assert S >= 1", "tract_assert P >= 0"],
    )


def reference(model, processor, wav, path):
    audio, sr = sf.read(wav, dtype="float32")
    assert sr == 16000 and audio.ndim == 1, "the sample must be 16 kHz mono"
    inputs = processor(audio, return_tensors="pt")
    tokens = model.generate(**inputs, do_sample=False)[0].tolist()
    path.write_text(json.dumps(tokens))
    print(processor.batch_decode([tokens], skip_special_tokens=True)[0])


def main(args):
    model_dir = args.assets / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    sample = args.sample
    if sample is None:
        sample = args.assets / "2086-149220-0033.wav"
        urllib.request.urlretrieve(SAMPLE_URL, sample)
    ckpt = snapshot_download(REPO, allow_patterns=["*.json", "model.safetensors"])
    processor = AutoProcessor.from_pretrained(ckpt)
    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(ckpt, dtype=torch.float32).eval()
    audio_config = processor.tokenizer.tokenizer.instruct_tokenizer.audio_encoder.audio_config
    delay_tokens = audio_config.get_num_delay_tokens(args.delay_ms)

    tekken = json.loads(Path(ckpt, "tekken.json").read_text())
    specials = len(tekken["special_tokens"])
    vocab = [base64.b64decode(v["token_bytes"]).hex() for v in tekken["vocab"][: model.config.text_config.vocab_size - specials]]
    (model_dir / "vocab.json").write_text(json.dumps(vocab))
    (model_dir / "config.json").write_text(
        json.dumps({"delay_tokens": delay_tokens, "left_pad_tokens": audio_config.n_left_pad_tokens, "special_tokens": specials})
    )

    export_encoder(model, processor, model_dir)
    export_decoder(model, delay_tokens, model_dir)
    if args.check:
        assert args.delay_ms == audio_config.transcription_delay_ms, "--check runs the default delay"
        (args.assets / "reference").mkdir(exist_ok=True)
        reference(model, processor, sample, args.assets / "reference" / "tokens.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("assets", type=Path)
    ap.add_argument("sample", type=Path, nargs="?")
    ap.add_argument("--delay-ms", type=int, default=480, help="transcription delay: a multiple of 80 ms")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    # torch-to-nnef walks the traced graph recursively, deeper than Python's default limit
    # over 32 layers: give it room, on a thread with a stack to match.
    sys.setrecursionlimit(1_000_000)
    threading.stack_size(1 << 30)
    failure = []

    def run():
        try:
            with torch.no_grad():
                main(args)
        except BaseException as e:
            failure.append(e)
            raise

    worker = threading.Thread(target=run)
    worker.start()
    worker.join()
    sys.exit(1 if failure else 0)
