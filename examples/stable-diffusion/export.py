#!/usr/bin/env python3
"""Export Stable Diffusion 1.5 components to ONNX."""

import os
import torch
import numpy as np
from pathlib import Path

ASSETS = Path("assets")
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def main():
    from diffusers import StableDiffusionPipeline

    component = os.environ.get("EXPORT_COMPONENT", "all")

    print(f"Loading {MODEL_ID}...")
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    pipe = pipe.to("cpu")

    ASSETS.mkdir(parents=True, exist_ok=True)

    if component in ("all", "vae_decoder"):
        print("Exporting vae_decoder...")
        vae = pipe.vae
        vae.eval()

        class VaeDecoder(torch.nn.Module):
            def __init__(self, vae):
                super().__init__()
                self.decoder = vae.decoder
                self.post_quant_conv = vae.post_quant_conv

            def forward(self, latent):
                latent = self.post_quant_conv(latent)
                return self.decoder(latent)

        vae_decoder = VaeDecoder(vae)
        latent = torch.randn(1, 4, 64, 64)
        with torch.no_grad():
            torch.onnx.export(
                vae_decoder,
                (latent,),
                str(ASSETS / "vae_decoder.onnx"),
                input_names=["latent"],
                output_names=["image"],
                opset_version=17,
                dynamo=False,
                dynamic_axes={"latent": {0: "batch"}, "image": {0: "batch"}},
            )
        print(f"  Exported to {ASSETS / 'vae_decoder.onnx'}")

    if component in ("all", "text_encoder"):
        print("Exporting text_encoder...")
        text_encoder = pipe.text_encoder
        text_encoder.eval()
        input_ids = torch.zeros(1, 77, dtype=torch.int64)
        with torch.no_grad():
            torch.onnx.export(
                text_encoder,
                (input_ids,),
                str(ASSETS / "text_encoder.onnx"),
                input_names=["input_ids"],
                output_names=["last_hidden_state", "pooler_output"],
                opset_version=17,
                dynamo=False,
                dynamic_axes={"input_ids": {0: "batch"}, "last_hidden_state": {0: "batch"}, "pooler_output": {0: "batch"}},
            )
        print(f"  Exported to {ASSETS / 'text_encoder.onnx'}")

    if component in ("all", "unet"):
        print("Exporting unet...")
        unet = pipe.unet
        unet.eval()
        sample = torch.randn(2, 4, 64, 64)
        timestep = torch.tensor([999, 999], dtype=torch.int64)
        encoder_hidden_states = torch.randn(2, 77, 768)
        with torch.no_grad():
            torch.onnx.export(
                unet,
                (sample, timestep, encoder_hidden_states),
                str(ASSETS / "unet.onnx"),
                input_names=["sample", "timestep", "encoder_hidden_states"],
                output_names=["noise_pred"],
                opset_version=17,
                dynamic_axes={
                    "sample": {0: "batch"},
                    "timestep": {0: "batch"},
                    "encoder_hidden_states": {0: "batch"},
                    "noise_pred": {0: "batch"},
                },
            )
        print(f"  Exported to {ASSETS / 'unet.onnx'}")

    # Save tokenizer for Rust side
    print("Saving tokenizer...")
    pipe.tokenizer.save_pretrained(str(ASSETS / "tokenizer"))

    print("Export complete.")


if __name__ == "__main__":
    main()
