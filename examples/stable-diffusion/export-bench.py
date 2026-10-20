#!/usr/bin/env python3
"""Export random-weight SD 1.5 UNet and VAE decoder to ONNX for the bench suite.

Same architecture as stable-diffusion-v1-5, randomly initialised, so the bench
models carry no checkpoint license. The UNet is fixed at batch 2 (one CFG step).
Convert each to the self-contained NNEF the bench suite loads:

    tract bench-assets/unet.onnx --nnef-tract-core --nnef-tract-transformers \
        dump --nnef-deterministic --nnef unet.nnef.tgz
"""

import torch
from pathlib import Path
from diffusers import AutoencoderKL, UNet2DConditionModel

OUT = Path("bench-assets")


def main():
    torch.manual_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)

    unet = UNet2DConditionModel(
        sample_size=64,
        in_channels=4,
        out_channels=4,
        block_out_channels=(320, 640, 1280, 1280),
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        layers_per_block=2,
        cross_attention_dim=768,
        attention_head_dim=8,
    ).eval()
    with torch.no_grad():
        torch.onnx.export(
            unet,
            (
                torch.randn(2, 4, 64, 64),
                torch.tensor([999, 999], dtype=torch.int64),
                torch.randn(2, 77, 768),
            ),
            str(OUT / "unet.onnx"),
            input_names=["sample", "timestep", "encoder_hidden_states"],
            output_names=["noise_pred"],
            opset_version=17,
        )

    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        latent_channels=4,
        sample_size=512,
    ).eval()

    class VaeDecoder(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.decoder = vae.decoder
            self.post_quant_conv = vae.post_quant_conv

        def forward(self, latent):
            return self.decoder(self.post_quant_conv(latent))

    with torch.no_grad():
        torch.onnx.export(
            VaeDecoder(vae),
            (torch.randn(1, 4, 64, 64),),
            str(OUT / "vae_decoder.onnx"),
            input_names=["latent"],
            output_names=["image"],
            opset_version=17,
        )


if __name__ == "__main__":
    main()
