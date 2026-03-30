#!/usr/bin/env python3
"""Export FLUX.1-schnell components to ONNX."""

import gc
import torch
from pathlib import Path
from transformers import CLIPTokenizerFast

ASSETS = Path("assets")
MODEL_ID = "black-forest-labs/FLUX.1-schnell"
DTYPE = torch.float16


class VaeDecoder(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent):
        return self.vae.decode(latent).sample


class FluxTransformerWrapper(torch.nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def _pack_latents(self, latents):
        # (B, C, H, W) → (B, H/2 * W/2, C*4)
        b, c, h, w = latents.shape
        latents = latents.reshape(b, c, h // 2, 2, w // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)  # (B, H/2, W/2, C, 2, 2)
        latents = latents.reshape(b, (h // 2) * (w // 2), c * 4)
        return latents

    def _unpack_latents(self, latents, h, w):
        # (B, H/2 * W/2, C*4) → (B, C, H, W)
        b, _, cd = latents.shape
        c = cd // 4
        latents = latents.reshape(b, h // 2, w // 2, c, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)  # (B, C, H/2, 2, W/2, 2)
        latents = latents.reshape(b, c, h, w)
        return latents

    def forward(self, latent, encoder_hidden_states, pooled_projections, timestep):
        b, c, h, w = latent.shape
        hidden_states = self._pack_latents(latent)
        seq_len = hidden_states.shape[1]
        txt_len = encoder_hidden_states.shape[1]

        # img_ids: (seq_len, 3) with (0, y, x) for each patch
        hy, wx = h // 2, w // 2
        img_ids = torch.zeros(seq_len, 3, device=latent.device, dtype=latent.dtype)
        ys = torch.arange(hy, device=latent.device, dtype=latent.dtype)
        xs = torch.arange(wx, device=latent.device, dtype=latent.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        img_ids[:, 1] = grid_y.flatten()
        img_ids[:, 2] = grid_x.flatten()
        img_ids = img_ids.unsqueeze(0).expand(b, -1, -1)

        # txt_ids: all zeros
        txt_ids = torch.zeros(b, txt_len, 3, device=latent.device, dtype=latent.dtype)

        out = self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep / 1000.0,
            img_ids=img_ids,
            txt_ids=txt_ids,
        ).sample

        return self._unpack_latents(out, h, w)


def free(*objects):
    for obj in objects:
        del obj
    gc.collect()


def load_component(cls, subfolder):
    return cls.from_pretrained(MODEL_ID, subfolder=subfolder, torch_dtype=DTYPE).to("cpu").eval()


def main():
    from diffusers import FluxTransformer2DModel, AutoencoderKL
    from transformers import CLIPTextModel, T5EncoderModel

    ASSETS.mkdir(parents=True, exist_ok=True)

    # --- Text Encoder (CLIP-L, hidden_size=768) ---
    print("Exporting text_encoder (CLIP-L)...")
    text_encoder = load_component(CLIPTextModel, "text_encoder")
    input_ids = torch.zeros(1, 77, dtype=torch.int64)
    with torch.no_grad():
        torch.onnx.export(
            text_encoder,
            (input_ids,),
            str(ASSETS / "text_encoder.onnx"),
            input_names=["input_ids"],
            output_names=["last_hidden_state", "text_embeds"],
            opset_version=18,
        )
    print(f"  Exported to {ASSETS / 'text_encoder.onnx'}")
    free(text_encoder)

    # --- Text Encoder 2 (T5-XXL, hidden_size=4096) ---
    print("Exporting text_encoder_2 (T5-XXL)...")
    text_encoder_2 = load_component(T5EncoderModel, "text_encoder_2")
    t5_input_ids = torch.zeros(1, 256, dtype=torch.int64)
    with torch.no_grad():
        torch.onnx.export(
            text_encoder_2,
            (t5_input_ids,),
            str(ASSETS / "text_encoder_2.onnx"),
            input_names=["input_ids"],
            output_names=["last_hidden_state"],
            opset_version=18,
        )
    print(f"  Exported to {ASSETS / 'text_encoder_2.onnx'}")
    free(text_encoder_2)

    # --- VAE Decoder (f32 — instance norm overflows in f16) ---
    print("Exporting vae_decoder (f32)...")
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32).to("cpu").eval()
    vae_decoder = VaeDecoder(vae)
    latent = torch.randn(1, 16, 128, 128, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            vae_decoder,
            (latent,),
            str(ASSETS / "vae_decoder.onnx"),
            input_names=["latent"],
            output_names=["image"],
            opset_version=18,
        )
    print(f"  Exported to {ASSETS / 'vae_decoder.onnx'}")
    free(vae, vae_decoder)

    # --- Transformer (Flux) ---
    print("Exporting transformer...")
    transformer = load_component(FluxTransformer2DModel, "transformer")
    wrapper = FluxTransformerWrapper(transformer)
    sample_latent = torch.randn(1, 16, 128, 128, dtype=DTYPE)
    sample_t5 = torch.randn(1, 256, 4096, dtype=DTYPE)
    sample_pooled = torch.randn(1, 768, dtype=DTYPE)
    sample_t = torch.tensor([500.0], dtype=DTYPE)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (sample_latent, sample_t5, sample_pooled, sample_t),
            str(ASSETS / "transformer.onnx"),
            input_names=["latent", "encoder_hidden_states", "pooled_projections", "timestep"],
            output_names=["output"],
            opset_version=18,
        )
    print(f"  Exported to {ASSETS / 'transformer.onnx'}")
    free(transformer, wrapper)

    # Save tokenizers
    print("Saving tokenizers...")
    tok = CLIPTokenizerFast.from_pretrained(MODEL_ID, subfolder="tokenizer")
    tok.save_pretrained(str(ASSETS / "tokenizer"))
    from transformers import AutoTokenizer
    tok2 = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer_2")
    tok2.save_pretrained(str(ASSETS / "tokenizer_2"))

    print("Export complete.")


if __name__ == "__main__":
    main()
