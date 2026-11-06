"""Export kyutai/mimi to NNEF as a streaming encoder and decoder, plus reference I/O.

- mimi-encoder: audio [1, 1, T] (24 kHz, T a multiple of 1920) -> codes [1, 8, T / 1920]
- mimi-decoder: codes [1, 8, F] -> audio [1, 1, 1920 * F]

Both graphs are written in plain torch over the HF weights so they trace cleanly
and pulsify: causal convs pad on the left only, attention uses the 250-frame
sliding window mask built from positions. The reference directory holds the
24 kHz input, the PyTorch codes and the PyTorch reconstruction.

Usage: python export.py <assets_dir> <sample.wav>
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
from torch_to_nnef import export_model_to_nnef
from torch_to_nnef.inference_target import TractNNEF
from transformers import MimiModel
from transformers.models.mimi.modeling_mimi import MimiConv1d, MimiConvTranspose1d, MimiResnetBlock

FRAME = 1920
CODEBOOKS = 8


def conv(x, mconv):
    c = mconv.conv
    k = (c.kernel_size[0] - 1) * c.dilation[0] + 1
    pad = k - c.stride[0]
    x = F.pad(x, (pad, 0), mode=mconv.pad_mode)
    return F.conv1d(x, c.weight, c.bias, c.stride, 0, c.dilation, c.groups)


def conv_tr(x, mconv):
    c = mconv.conv
    trim = c.kernel_size[0] - c.stride[0]
    y = F.conv_transpose1d(x, c.weight, c.bias, c.stride, 0, 0, c.groups)
    return y[..., :-trim] if trim > 0 else y


def seanet(x, layers):
    for layer in layers:
        if isinstance(layer, MimiConv1d):
            x = conv(x, layer)
        elif isinstance(layer, MimiConvTranspose1d):
            x = conv_tr(x, layer)
        elif isinstance(layer, MimiResnetBlock):
            y = x
            for sub in layer.block:
                y = conv(y, sub) if isinstance(sub, MimiConv1d) else F.elu(y)
            x = x + y
        else:
            x = F.elu(x)
    return x


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Transformer(torch.nn.Module):
    def __init__(self, hf, config):
        super().__init__()
        self.layers = hf.layers
        self.heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.window = config.sliding_window
        theta = config.rope_parameters["rope_theta"]
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # x: [B, C, T] -> [B, C, T]
        x = x.transpose(1, 2)
        t = x.shape[1]
        pos = torch.arange(t).to(torch.float32)
        freqs = pos[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos(), emb.sin()
        idx = torch.arange(t)
        diff = idx[:, None] - idx[None, :]
        mask = (diff >= 0) & (diff < self.window)
        for layer in self.layers:
            h = layer.input_layernorm(x)
            att = layer.self_attn
            q = att.q_proj(h).view(1, t, self.heads, self.head_dim).transpose(1, 2)
            k = att.k_proj(h).view(1, t, self.heads, self.head_dim).transpose(1, 2)
            v = att.v_proj(h).view(1, t, self.heads, self.head_dim).transpose(1, 2)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            scores = (q @ k.transpose(2, 3)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~mask, float("-inf"))
            o = scores.softmax(-1) @ v
            o = o.transpose(1, 2).reshape(1, t, self.heads * self.head_dim)
            x = x + layer.self_attn_layer_scale(att.o_proj(o))
            h = layer.post_attention_layernorm(x)
            x = x + layer.mlp_layer_scale(layer.mlp.fc2(F.gelu(layer.mlp.fc1(h))))
        return x.transpose(1, 2)


def codebooks(mimi, num_codebooks):
    def embed(layer):
        cb = layer.codebook
        return cb.embed_sum / cb.cluster_usage.clamp(min=cb.epsilon)[:, None]

    q = mimi.quantizer
    books = [embed(l) for l in q.semantic_residual_vector_quantizer.layers]
    books += [embed(l) for l in q.acoustic_residual_vector_quantizer.layers[: num_codebooks - 1]]
    return torch.stack(books)  # [K, N, D]


class Encoder(torch.nn.Module):
    def __init__(self, mimi: MimiModel, num_codebooks: int):
        super().__init__()
        self.seanet = mimi.encoder.encoder_seanet.layers
        self.transformer = Transformer(mimi.encoder.encoder_transformer, mimi.config)
        self.downsample = mimi.encoder.downsample
        self.semantic_proj = mimi.quantizer.semantic_residual_vector_quantizer.input_proj
        self.acoustic_proj = mimi.quantizer.acoustic_residual_vector_quantizer.input_proj
        self.register_buffer("books", codebooks(mimi, num_codebooks))
        self.register_buffer("books_sq", (self.books**2).sum(-1))  # [K, N]

    def rvq(self, x, proj, first, count):
        r = proj(x).transpose(1, 2)[0]  # [T, D]
        codes = []
        for i in range(first, first + count):
            book = self.books[i]
            dist = self.books_sq[i][None, :] - 2 * r @ book.T
            idx = dist.argmin(-1)
            codes.append(idx)
            r = r - book[idx]
        return codes

    def forward(self, audio):
        x = seanet(audio, self.seanet)
        x = self.transformer(x)
        x = conv(x, self.downsample)
        k = self.books.shape[0]
        codes = self.rvq(x, self.semantic_proj, 0, 1)
        codes += self.rvq(x, self.acoustic_proj, 1, k - 1)
        return torch.stack(codes)[None]  # [1, K, T]


class Decoder(torch.nn.Module):
    def __init__(self, mimi: MimiModel, num_codebooks: int):
        super().__init__()
        self.semantic_proj = mimi.quantizer.semantic_residual_vector_quantizer.output_proj
        self.acoustic_proj = mimi.quantizer.acoustic_residual_vector_quantizer.output_proj
        self.upsample = mimi.decoder.upsample
        self.transformer = Transformer(mimi.decoder.decoder_transformer, mimi.config)
        self.seanet = mimi.decoder.decoder_seanet.layers
        self.register_buffer("books", codebooks(mimi, num_codebooks))

    def forward(self, codes):
        sem = self.books[0][codes[0, 0]]
        aco = self.books[1][codes[0, 1]]
        for i in range(2, self.books.shape[0]):
            aco = aco + self.books[i][codes[0, i]]
        x = self.semantic_proj(sem.T[None]) + self.acoustic_proj(aco.T[None])
        x = conv_tr(x, self.upsample)
        x = self.transformer(x)
        return seanet(x, self.seanet)


def target():
    tract = os.environ.get("TRACT_BIN")
    if tract is None:
        return TractNNEF.latest()
    version = subprocess.check_output([tract, "--version"], text=True).split()[-1]
    return TractNNEF(version, specific_tract_binary_path=Path(tract))


def main(assets: Path, wav: Path):
    model_dir, ref_dir = assets / "model", assets / "reference"
    model_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)
    mimi = MimiModel.from_pretrained("kyutai/mimi").eval()
    signal, sr = sf.read(wav, dtype="float32")
    assert signal.ndim == 1
    signal = torch.from_numpy(signal)[None, None]
    if sr != 24000:
        signal = F.interpolate(signal, scale_factor=24000 / sr, mode="linear", align_corners=False)
    audio = signal[..., : signal.shape[-1] // FRAME * FRAME].contiguous()

    encoder, decoder = Encoder(mimi, CODEBOOKS).eval(), Decoder(mimi, CODEBOOKS).eval()
    with torch.no_grad():
        codes = encoder(audio)
        assert (codes == mimi.encode(audio, num_quantizers=CODEBOOKS).audio_codes).all()
        recon = decoder(codes)
    sf.write(ref_dir / "input.wav", audio[0, 0].numpy(), 24000, subtype="FLOAT")
    sf.write(ref_dir / "recon.wav", recon[0, 0].numpy(), 24000, subtype="FLOAT")
    (ref_dir / "codes.json").write_text(json.dumps(codes[0].tolist()))

    it = target()
    it.dynamic_axes = {"audio": {2: "T"}}
    export_model_to_nnef(
        model=encoder,
        args=(audio,),
        file_path_export=model_dir / "mimi-encoder.nnef.tgz",
        input_names=["audio"],
        output_names=["codes"],
        inference_target=it,
        custom_extensions=[f"tract_assert T >= {FRAME}"],
    )
    it = target()
    it.dynamic_axes = {"codes": {2: "F"}}
    export_model_to_nnef(
        model=decoder,
        args=(codes,),
        file_path_export=model_dir / "mimi-decoder.nnef.tgz",
        input_names=["codes"],
        output_names=["audio"],
        inference_target=it,
        custom_extensions=["tract_assert F >= 1"],
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
