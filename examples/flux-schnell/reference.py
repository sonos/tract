#!/usr/bin/env python3
"""Generate reference I/O data for FLUX.1-schnell model validation."""

import gc
import torch
import numpy as np
from pathlib import Path

ASSETS = Path("assets")
MODEL_ID = "black-forest-labs/FLUX.1-schnell"
PROMPT = "a photo of a cat"
SEED = 42
DTYPE = torch.float16
GPU = "cuda" if torch.cuda.is_available() else "cpu"


def free(*objects):
    for obj in objects:
        del obj
    gc.collect()
    if GPU == "cuda":
        torch.cuda.empty_cache()


def load_component(cls, subfolder, device="cpu"):
    return cls.from_pretrained(MODEL_ID, subfolder=subfolder, torch_dtype=DTYPE).to(device).eval()


def main():
    from transformers import CLIPTextModel, T5EncoderModel, CLIPTokenizerFast, AutoTokenizer
    from diffusers import AutoencoderKL, FluxTransformer2DModel

    # --- Tokenize ---
    print("Tokenizing...")
    tokenizer = CLIPTokenizerFast.from_pretrained(MODEL_ID, subfolder="tokenizer")
    tokens = tokenizer(
        PROMPT, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    )
    input_ids = tokens.input_ids  # (1, 77)

    tokenizer_2 = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer_2")
    t5_tokens = tokenizer_2(
        PROMPT, padding="max_length", max_length=256, truncation=True, return_tensors="pt"
    )
    t5_input_ids = t5_tokens.input_ids  # (1, 256)
    free(tokenizer, tokenizer_2)

    # --- Text Encoder 1 (CLIP-L) — CPU (small, fast) ---
    print("Running CLIP-L text encoder (CPU)...")
    text_encoder = load_component(CLIPTextModel, "text_encoder")
    with torch.no_grad():
        te1_out = text_encoder(input_ids, output_hidden_states=False)

    pooled = te1_out.pooler_output  # (1, 768)
    np.savez(
        str(ASSETS / "text_encoder.io.npz"),
        input_ids=input_ids.numpy(),
        text_embeds=pooled.float().numpy(),
        last_hidden_state=te1_out.last_hidden_state.float().numpy(),
    )
    print("Saved text_encoder.io.npz")
    pooled = pooled.clone()
    free(text_encoder, te1_out)

    # --- Text Encoder 2 (T5-XXL) — CPU (large but tractable) ---
    print("Running T5-XXL text encoder (CPU)...")
    text_encoder_2 = load_component(T5EncoderModel, "text_encoder_2")
    with torch.no_grad():
        te2_out = text_encoder_2(t5_input_ids)

    t5_embeds = te2_out.last_hidden_state  # (1, 256, 4096)
    np.savez(
        str(ASSETS / "text_encoder_2.io.npz"),
        input_ids=t5_input_ids.numpy(),
        last_hidden_state=t5_embeds.float().numpy(),
    )
    print("Saved text_encoder_2.io.npz")
    t5_embeds = t5_embeds.clone()
    del text_encoder_2, te2_out
    gc.collect()

    # --- Transformer (one step) — GPU (12B params, too slow on CPU) ---
    print("Running transformer (single step, GPU)...")
    from export import FluxTransformerWrapper

    transformer = load_component(FluxTransformer2DModel, "transformer", device=GPU)
    wrapper = FluxTransformerWrapper(transformer)
    wrapper.eval()

    generator = torch.Generator().manual_seed(SEED)
    latent = torch.randn(1, 16, 128, 128, generator=generator, dtype=DTYPE)

    # Schnell uses shift=1.0 so sigmas are simply linear: 1.0, 0.75, 0.5, 0.25, 0.0
    # timestep[0] = sigma[0] * 1000 = 1000.0
    timestep = torch.tensor([1000.0], dtype=DTYPE)

    with torch.no_grad():
        pred = wrapper(
            latent.to(GPU), t5_embeds.to(GPU), pooled.to(GPU), timestep.to(GPU)
        )

    pred_cpu = pred.cpu()
    np.savez(
        str(ASSETS / "transformer.io.npz"),
        latent=latent.numpy(),
        encoder_hidden_states=t5_embeds.numpy(),
        pooled_projections=pooled.numpy(),
        timestep=timestep.numpy(),
        output=pred_cpu.numpy(),
    )
    print("Saved transformer.io.npz")
    free(transformer, wrapper, pred, pred_cpu, t5_embeds, pooled)

    # --- VAE Decoder — CPU (small, fast) ---
    print("Running VAE decoder (CPU)...")
    vae = load_component(AutoencoderKL, "vae")

    scaling_factor = 0.3611
    shift_factor = 0.1159
    latent_scaled = latent / scaling_factor + shift_factor

    with torch.no_grad():
        image = vae.decode(latent_scaled).sample

    np.savez(
        str(ASSETS / "vae_decoder.io.npz"),
        latent=latent_scaled.numpy(),
        image=image.numpy(),
    )
    print("Saved vae_decoder.io.npz")
    free(vae)

    print("Reference data complete.")


if __name__ == "__main__":
    main()
