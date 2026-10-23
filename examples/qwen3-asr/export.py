"""Export Qwen3-ASR-0.6B to NNEF as three graphs and a tokenizer.

- preprocessor: 16 kHz signal [1, N] -> Whisper log-mel [128, T]
- encoder: one 8 s attention window, chunks [S, 128, 100] + key_bias [S, 13] -> [S * 13, 1024]
- decoder: Qwen3 LM with KV cache, audio embeddings spliced at the audio pad tokens

Usage: python export.py <export_dir> <sample.wav>
"""

import os
import sys
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
from qwen_asr.core.transformers_backend import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)
from torch_to_nnef import export_model_to_nnef
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.inference_target.tract import TractCheckTolerance
from torch_to_nnef_llm.exporter import LLMExporter
from torch_to_nnef_llm.models.handlers.base import IOSpec, reset_special_ids_to_filler
from torch_to_nnef_llm.models.handlers.registry import register_handler
from torch_to_nnef_llm.models.handlers.voxtral import VoxtralArchitectureHandler
from transformers import AutoConfig, AutoModel, AutoProcessor

MODEL = os.environ.get("QWEN3_ASR_MODEL", "Qwen/Qwen3-ASR-0.6B")
CHUNK = 100
TOK_PER_CHUNK = 13
N_AUDIO = 25


class Preprocessor(torch.nn.Module):
    """WhisperFeatureExtractor's torch path, for a single unpadded signal."""

    def __init__(self, feature_extractor):
        super().__init__()
        self.n_fft = feature_extractor.n_fft
        self.hop = feature_extractor.hop_length
        self.register_buffer("window", torch.hann_window(self.n_fft))
        self.register_buffer(
            "mel_filters", torch.from_numpy(feature_extractor.mel_filters).float().T.contiguous()
        )

    def forward(self, signal):
        stft = torch.stft(signal[0], self.n_fft, self.hop, window=self.window, return_complex=True)
        power = stft[..., :-1].abs() ** 2
        log_spec = torch.clamp(self.mel_filters @ power, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        return (log_spec + 4.0) / 4.0


class AudioEncoder(torch.nn.Module):
    """The audio tower over one attention window of S one-second chunks.

    Tokens of the zero-padded tail chunk are masked out of attention by key_bias
    (0 or -inf); the caller drops their rows from the output.
    """

    def __init__(self, tower):
        super().__init__()
        self.t = tower
        self.register_buffer(
            "pos", tower.positional_embedding.positional_embedding[:TOK_PER_CHUNK].clone()
        )

    def forward(self, chunks, key_bias):
        t = self.t
        x = chunks.unsqueeze(1)
        x = F.gelu(t.conv2d1(x))
        x = F.gelu(t.conv2d2(x))
        x = F.gelu(t.conv2d3(x))
        s, c, f, n = x.shape
        x = t.conv_out(x.permute(0, 3, 1, 2).reshape(s, n, c * f)) + self.pos
        x = x.reshape(s * n, -1)
        bias = key_bias.reshape(1, 1, 1, -1)
        for layer in t.layers:
            a = layer.self_attn
            h = layer.self_attn_layer_norm(x)
            q, k, v = (
                p(h).reshape(-1, a.num_heads, a.head_dim).transpose(0, 1).unsqueeze(0)
                for p in (a.q_proj, a.k_proj, a.v_proj)
            )
            w = torch.softmax(q @ k.transpose(-1, -2) * a.scaling + bias, dim=-1)
            x = x + a.out_proj((w @ v).squeeze(0).transpose(0, 1).reshape(-1, a.embed_dim))
            x = x + layer.fc2(F.gelu(layer.fc1(layer.final_layer_norm(x))))
        x = t.ln_post(x)
        return t.proj2(F.gelu(t.proj1(x)))


@register_handler
class Qwen3ASRDecoderHandler(VoxtralArchitectureHandler):
    """Qwen3 decoder fed with audio embeddings at audio_token_id; keeps last logits only."""

    ARCH_NAMES = ("qwen3_asr",)

    def build_input_spec(
        self,
        *,
        tokenizer,
        config_helper,
        inputs_dtype,
        sample_text,
        n_input_tokens,
        n_past_input_tokens,
        real_kv_cache=None,
    ):
        in_names, out_names, pkv, dyn = config_helper.build_kv_cache_infos(
            n_past_input_tokens=n_past_input_tokens,
            force_inputs_dtype=inputs_dtype,
            real_kv_cache=real_kv_cache,
        )
        conf = config_helper.decoder_conf
        audio_token_id = config_helper.conf.audio_token_id
        input_ids = torch.randint(0, conf.vocab_size, (1, max(n_input_tokens, N_AUDIO + 2)))
        reset_special_ids_to_filler(input_ids, {audio_token_id}, conf.vocab_size)
        input_ids[:, 1 : 1 + N_AUDIO] = audio_token_id
        # real audio embeddings are ~0.02 std: unit-scale noise blows the check tolerance
        audio = torch.randn((N_AUDIO, conf.hidden_size), dtype=inputs_dtype) * 0.02
        return IOSpec(
            inputs=(input_ids, *pkv, audio),
            input_names=["input_ids"] + in_names + self.STATE_INPUT_NAMES,
            output_names=["outputs"] + out_names + self.STATE_OUTPUT_NAMES,
            dynamic_axes={**dyn, "in_audio_embeddings": {0: "AUD"}},
        )

    def build_forward_outputs(self, *, model, model_outputs, state_context, num_logits_to_keep):
        outputs = super().build_forward_outputs(
            model=model,
            model_outputs=model_outputs,
            state_context=state_context,
            num_logits_to_keep=num_logits_to_keep,
        )
        return [outputs[0][:, -num_logits_to_keep:]] + outputs[1:]


def main(export_dir: Path, wav: Path):
    AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
    AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
    AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float32).eval()
    export_dir.mkdir(parents=True, exist_ok=True)
    target = TractNNEF.latest

    signal, sr = sf.read(wav, dtype="float32")
    assert sr == 16000 and signal.ndim == 1
    signal = torch.from_numpy(signal)[None]
    pre = Preprocessor(proc.feature_extractor)
    export_model_to_nnef(
        model=pre,
        args=(signal,),
        file_path_export=export_dir / "preprocessor.nnef.tgz",
        input_names=["signal"],
        output_names=["features"],
        inference_target=target(),
    )

    with torch.no_grad():
        mel = pre(signal)[:, : CHUNK * 3]
    chunks = mel.reshape(128, 3, CHUNK).permute(1, 0, 2).contiguous()
    key_bias = torch.zeros(3, TOK_PER_CHUNK)
    key_bias[2, 6:] = torch.finfo(torch.float32).min
    it = target()
    it.dynamic_axes = {"chunks": {0: "S"}, "key_bias": {0: "S"}}
    export_model_to_nnef(
        model=AudioEncoder(model.thinker.audio_tower).eval(),
        args=(chunks, key_bias),
        file_path_export=export_dir / "encoder.nnef.tgz",
        input_names=["chunks", "key_bias"],
        output_names=["audio_embeddings"],
        inference_target=it,
    )

    exporter = LLMExporter(model.thinker, proc.tokenizer, force_module_dtype="f32")
    exporter.dump_with_inference_target(
        # f32 drift over 28 layers exceeds the default LM tolerance by ~2e-4
        inference_target=exporter.build_inference_target(
            tract_check_io_tolerance=TractCheckTolerance.VERY
        ),
        export_dirpath=export_dir / "decoder",
        ignore_already_exist_dir=True,
    )
    proc.tokenizer.backend_tokenizer.save(str(export_dir / "tokenizer.json"))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
