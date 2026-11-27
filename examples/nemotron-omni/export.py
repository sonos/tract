"""Export NVIDIA's Nemotron 3 Nano Omni (nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16) to
NNEF for the nemotron-omni example.

Writes, under <assets>:
- model/: torch-to-nnef's multimodal export. decoder/model.nnef.tgz is the language model (f16,
  routed experts in Q4_0) with its Mamba-2 and attention states as explicit inputs and outputs;
  vision/model.nnef.tgz turns one image's patches into its embeddings, audio/model.nnef.tgz one
  clip's log-mel frames into theirs; multimodal.json describes how they plug together.
- mel.nnef.tgz: the audio feature extractor, 16 kHz samples [1, N] to normalized log-mel frames
  [1, N / 160, 128], as transformers' ParakeetFeatureExtractor computes them for one clip.
- tokenizer.json.

Usage: python export.py <assets> [--only mel|model] [--checkpoint DIR]
"""

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from torch import nn
from transformers.audio_utils import mel_filter_bank

REPO = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
TRACT_VERSION = "0.23.11"

SAMPLE_RATE = 16000
N_FFT = 512
WIN = 400
HOP = 160
MELS = 128
PREEMPH = 0.97
GUARD = 2.0**-24
EPSILON = 1e-5


def checkpoint_view(assets, checkpoint):
    """The checkpoint as transformers' native nemotron_h_omni reads it: NVIDIA ships a config for
    its remote code, and RADIO's layer scales (all ones) are not in the weights."""
    src = checkpoint or Path(snapshot_download(REPO))
    view = assets / "checkpoint"
    view.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name not in ("config.json", "model.safetensors.index.json") and not (view / f.name).exists():
            (view / f.name).symlink_to(f)
    nv = json.loads((src / "config.json").read_text())
    text = {k: v for k, v in nv["llm_config"].items() if k not in ("auto_map", "architectures")}
    text["model_type"] = "nemotron_h"
    audio = dict(nv["sound_config"], model_type="parakeet_encoder")
    config = {
        "architectures": ["NemotronH_Omni_Reasoning_V3"],
        "model_type": "nemotron_h_omni",
        "text_config": text,
        "vision_config": {"model_type": "radio", "video_temporal_patch_size": 2},
        "audio_config": audio,
        "downsample_ratio": nv["downsample_ratio"],
        "projector_hidden_size": nv["projector_hidden_size"],
        "vision_hidden_size": nv["vit_hidden_size"],
        "image_token_id": nv["img_context_token_id"],
        "video_token_id": nv["video_context_token_id"],
        "audio_token_id": nv["sound_context_token_id"],
        "video_temporal_patch_size": 2,
        "torch_dtype": "bfloat16",
    }
    (view / "config.json").write_text(json.dumps(config, indent=1))
    index = json.loads((src / "model.safetensors.index.json").read_text())
    blocks = "vision_model.radio_model.model.blocks."
    layers = 1 + max(int(k[len(blocks) :].split(".")[0]) for k in index["weight_map"] if k.startswith(blocks))
    scales = {
        f"vision_model.encoder.layer.{layer}.layer_scale{which}.lambda1": torch.ones(nv["vit_hidden_size"], dtype=torch.bfloat16)
        for layer in range(layers)
        for which in (1, 2)
    }
    save_file(scales, view / "radio-layerscale.safetensors", metadata={"format": "pt"})
    index["weight_map"].update({k: "radio-layerscale.safetensors" for k in scales})
    (view / "model.safetensors.index.json").write_text(json.dumps(index))
    return view


def export_model(view, out):
    from torch_to_nnef_llm.exporter import LM_VAR_SCHEME
    from torch_to_nnef_llm.multimodal_exporter import MultiModalExporter

    exporter = MultiModalExporter.load(
        local_dir=view,
        force_module_dtype="f16",
        device_map="t2n_offload_disk",
        trust_remote_code=False,
    )
    exporter.decoder_exporter.mark_moe_experts_for_q40(layout="linear")
    target = exporter.decoder_exporter.build_inference_target(
        tract_specific_version=TRACT_VERSION, no_verify=True
    )
    exporter.export(out, target, LM_VAR_SCHEME, no_verify=True, check_inference_modes=False)


class Mel(nn.Module):
    """ParakeetFeatureExtractor on one clip: preemphasis, centered zero-padded STFT, power,
    Slaney mel, log, then per-bin normalization over the clip's frames."""

    def __init__(self):
        super().__init__()
        window = torch.hann_window(WIN, periodic=False, dtype=torch.float64)
        window = F.pad(window, ((N_FFT - WIN) // 2, (N_FFT - WIN + 1) // 2))
        k = torch.arange(N_FFT // 2 + 1, dtype=torch.float64)[:, None]
        t = torch.arange(N_FFT, dtype=torch.float64)[None]
        angle = 2 * math.pi * k * t / N_FFT
        basis = torch.cat([torch.cos(angle) * window, -torch.sin(angle) * window])
        self.register_buffer("basis", basis[:, None].float())
        fb = mel_filter_bank(
            num_frequency_bins=N_FFT // 2 + 1,
            num_mel_filters=MELS,
            min_frequency=0.0,
            max_frequency=SAMPLE_RATE / 2,
            sampling_rate=SAMPLE_RATE,
            norm="slaney",
            mel_scale="slaney",
        )
        self.register_buffer("fb", torch.from_numpy(fb.T.astype(np.float32)))

    def forward(self, samples):
        x = torch.cat([samples[:, :1], samples[:, 1:] - PREEMPH * samples[:, :-1]], dim=1)
        frames = x.shape[1] // HOP
        x = F.pad(x, (N_FFT // 2, N_FFT // 2))
        re, im = F.conv1d(x[:, None], self.basis, stride=HOP).chunk(2, dim=1)
        power = re * re + im * im
        mel = torch.log(torch.matmul(self.fb, power[:, :, :frames]) + GUARD)
        mean = mel.mean(dim=2, keepdim=True)
        var = ((mel - mean) ** 2).sum(dim=2, keepdim=True) / (frames - 1)
        return ((mel - mean) / (var.sqrt() + EPSILON)).transpose(1, 2)


def export_mel(out):
    from torch_to_nnef import export_model_to_nnef
    from torch_to_nnef.inference_target import TractNNEF
    from torch_to_nnef.utils import SemanticVersion

    target = TractNNEF(SemanticVersion.from_str(TRACT_VERSION), dynamic_axes={"samples": {1: "N"}}, check_io=False)
    with torch.no_grad():
        export_model_to_nnef(
            model=Mel().eval(),
            args=(torch.randn(1, 3 * SAMPLE_RATE),),
            file_path_export=out / "mel.nnef.tgz",
            input_names=["samples"],
            output_names=["mel"],
            inference_target=target,
            custom_extensions=["tract_assert N >= 320"],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("assets", type=Path)
    parser.add_argument("--only", choices=["mel", "model"])
    parser.add_argument("--checkpoint", type=Path, help=f"a local copy of {REPO} (default: download it)")
    args = parser.parse_args()
    args.assets.mkdir(parents=True, exist_ok=True)
    if args.only != "model":
        export_mel(args.assets)
    if args.only != "mel":
        view = checkpoint_view(args.assets, args.checkpoint)
        shutil.copy(view / "tokenizer.json", args.assets / "tokenizer.json")
        # torch-to-nnef offloads the weights to a temporary copy on disk: keep it next to the
        # assets and remove it whatever happens
        with tempfile.TemporaryDirectory(dir=args.assets) as tmp:
            os.environ["TMPDIR"] = tempfile.tempdir = tmp
            # and the quantized experts too, one at a time, instead of holding them all in RAM
            os.environ["T2N_OFFLOAD_QUANTIZED_EXPERTS"] = "1"
            export_model(view, args.assets / "model")


if __name__ == "__main__":
    # tracing the 52-layer decoder recurses deeply
    sys.setrecursionlimit(1_000_000)
    threading.stack_size(1 << 30)
    failure = []

    def run():
        try:
            main()
        except BaseException as e:
            failure.append(e)
            raise

    worker = threading.Thread(target=run)
    worker.start()
    worker.join()
    sys.exit(1 if failure else 0)
