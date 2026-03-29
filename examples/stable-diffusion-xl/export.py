#!/usr/bin/env python3
"""Export Stable Diffusion XL 1.0 components to ONNX."""

import os
import torch
from pathlib import Path
from transformers import CLIPTokenizerFast

ASSETS = Path("assets")
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


def main():
    from diffusers import StableDiffusionXLPipeline

    print(f"Loading {MODEL_ID}...")
    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float32, variant="fp16")
    pipe = pipe.to("cpu")

    ASSETS.mkdir(parents=True, exist_ok=True)

    # --- Text Encoder 1 (CLIP ViT-L/14, hidden_size=768) ---
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
        )
    print(f"  Exported to {ASSETS / 'text_encoder.onnx'}")

    # --- Text Encoder 2 (OpenCLIP ViT-bigG, hidden_size=1280) ---
    print("Exporting text_encoder_2...")
    text_encoder_2 = pipe.text_encoder_2
    text_encoder_2.eval()
    with torch.no_grad():
        torch.onnx.export(
            text_encoder_2,
            (input_ids,),
            str(ASSETS / "text_encoder_2.onnx"),
            input_names=["input_ids"],
            output_names=["last_hidden_state", "text_embeds"],
            opset_version=17,
        )
    print(f"  Exported to {ASSETS / 'text_encoder_2.onnx'}")

    # --- VAE Decoder ---
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
    latent = torch.randn(1, 4, 128, 128)
    with torch.no_grad():
        torch.onnx.export(
            vae_decoder,
            (latent,),
            str(ASSETS / "vae_decoder.onnx"),
            input_names=["latent"],
            output_names=["image"],
            opset_version=17,
        )
    print(f"  Exported to {ASSETS / 'vae_decoder.onnx'}")

    # --- UNet (with added_cond_kwargs) ---
    print("Exporting unet...")
    unet = pipe.unet
    unet.eval()
    sample = torch.randn(2, 4, 128, 128)
    timestep = torch.tensor([999, 999], dtype=torch.int64)
    encoder_hidden_states = torch.randn(2, 77, 2048)
    # SDXL added conditions
    time_ids = torch.zeros(2, 6)  # [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
    text_embeds = torch.randn(2, 1280)  # pooled from text_encoder_2

    class UNetWrapper(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet

        def forward(self, sample, timestep, encoder_hidden_states, time_ids, text_embeds):
            return self.unet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs={"time_ids": time_ids, "text_embeds": text_embeds},
            ).sample

    wrapper = UNetWrapper(unet)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (sample, timestep, encoder_hidden_states, time_ids, text_embeds),
            str(ASSETS / "unet.onnx"),
            input_names=["sample", "timestep", "encoder_hidden_states", "time_ids", "text_embeds"],
            output_names=["noise_pred"],
            opset_version=17,
            dynamic_axes={
                "sample": {0: "batch"},
                "timestep": {0: "batch"},
                "encoder_hidden_states": {0: "batch"},
                "time_ids": {0: "batch"},
                "text_embeds": {0: "batch"},
                "noise_pred": {0: "batch"},
            },
        )
    print(f"  Exported to {ASSETS / 'unet.onnx'}")

    # Save fast tokenizer (same for both text encoders in SDXL)
    print("Saving tokenizer...")
    tok = CLIPTokenizerFast.from_pretrained(MODEL_ID, subfolder="tokenizer")
    tok.save_pretrained(str(ASSETS / "tokenizer"))

    print("Export complete.")


if __name__ == "__main__":
    main()
