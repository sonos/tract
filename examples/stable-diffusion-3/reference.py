#!/usr/bin/env python3
"""Generate reference I/O data for SD3 model validation."""

import torch
import numpy as np
from pathlib import Path
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler

ASSETS = Path("assets")
MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
PROMPT = "a photo of a cat"
SEED = 42
NUM_STEPS = 28


def main():
    print(f"Loading {MODEL_ID} (without T5)...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, text_encoder_3=None, tokenizer_3=None,
    )
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cpu")

    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2

    # Tokenize
    tokens = tokenizer(PROMPT, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    uncond_tokens = tokenizer("", padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    tokens_2 = tokenizer_2(PROMPT, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    uncond_tokens_2 = tokenizer_2("", padding="max_length", max_length=77, truncation=True, return_tensors="pt")

    input_ids = tokens.input_ids
    uncond_input_ids = uncond_tokens.input_ids
    input_ids_2 = tokens_2.input_ids
    uncond_input_ids_2 = uncond_tokens_2.input_ids

    # --- Text Encoder 1 (CLIP-L) ---
    with torch.no_grad():
        te1_cond = pipe.text_encoder(input_ids, output_hidden_states=True)
        te1_uncond = pipe.text_encoder(uncond_input_ids, output_hidden_states=True)

    # CLIPTextModelWithProjection: text_embeds = projected pooled, hidden_states[-2] = penultimate
    np.savez(str(ASSETS / "text_encoder.io.npz"),
        input_ids=input_ids.numpy(),
        text_embeds=te1_cond.text_embeds.numpy(),
        last_hidden_state=te1_cond.last_hidden_state.numpy(),
    )
    print("Saved text_encoder.io.npz")

    # --- Text Encoder 2 (CLIP-G) ---
    with torch.no_grad():
        te2_cond = pipe.text_encoder_2(input_ids_2, output_hidden_states=True)
        te2_uncond = pipe.text_encoder_2(uncond_input_ids_2, output_hidden_states=True)

    np.savez(str(ASSETS / "text_encoder_2.io.npz"),
        input_ids=input_ids_2.numpy(),
        text_embeds=te2_cond.text_embeds.numpy(),
        last_hidden_state=te2_cond.last_hidden_state.numpy(),
    )
    print("Saved text_encoder_2.io.npz")

    # --- Concat text embeddings (without T5) ---
    # Penultimate hidden states from both CLIPs
    cond_h1 = te1_cond.hidden_states[-2]   # (1, 77, 768)
    cond_h2 = te2_cond.hidden_states[-2]   # (1, 77, 1280)
    uncond_h1 = te1_uncond.hidden_states[-2]
    uncond_h2 = te2_uncond.hidden_states[-2]

    # Concatenate along feature dim then pad to 4096
    cond_clip = torch.cat([cond_h1, cond_h2], dim=-1)      # (1, 77, 2048)
    uncond_clip = torch.cat([uncond_h1, uncond_h2], dim=-1)
    cond_embeds = torch.nn.functional.pad(cond_clip, (0, 4096 - 2048))    # (1, 77, 4096)
    uncond_embeds = torch.nn.functional.pad(uncond_clip, (0, 4096 - 2048))

    # Pooled: concatenate CLIP pooled outputs
    cond_pooled = torch.cat([te1_cond.text_embeds, te2_cond.text_embeds], dim=-1)      # (1, 2048)
    uncond_pooled = torch.cat([te1_uncond.text_embeds, te2_uncond.text_embeds], dim=-1)

    print(f"Text embeddings: {cond_embeds.shape}, pooled: {cond_pooled.shape}")

    # --- Transformer (one step) ---
    pipe.scheduler.set_timesteps(NUM_STEPS)
    ts = pipe.scheduler.timesteps
    sigmas = pipe.scheduler.sigmas
    generator = torch.Generator().manual_seed(SEED)
    latent = torch.randn(1, 16, 128, 128, generator=generator)
    # Scale initial noise by first sigma
    latent = latent * sigmas[0]

    t0 = ts[0]
    sigma0 = sigmas[0]
    lat_in = latent / ((sigma0**2 + 1) ** 0.5)

    with torch.no_grad():
        pred = pipe.transformer(
            hidden_states=lat_in,
            encoder_hidden_states=cond_embeds,
            pooled_projections=cond_pooled,
            timestep=t0.unsqueeze(0),
        ).sample

    np.savez(str(ASSETS / "transformer.io.npz"),
        hidden_states=lat_in.numpy(),
        encoder_hidden_states=cond_embeds.numpy(),
        pooled_projections=cond_pooled.numpy(),
        timestep=np.array([t0.item()], dtype=np.float32),
        output=pred.numpy(),
    )
    print("Saved transformer.io.npz")

    # --- Full denoising loop ---
    print(f"Running full pipeline ({NUM_STEPS} steps)...")
    latent = torch.randn(1, 16, 128, 128, generator=torch.Generator().manual_seed(SEED))
    latent = latent * sigmas[0]

    with torch.no_grad():
        for i, t in enumerate(ts):
            lat_uncond = latent.clone()
            lat_cond = latent.clone()
            lat_in = torch.cat([lat_uncond, lat_cond])

            pred = pipe.transformer(
                hidden_states=lat_in,
                encoder_hidden_states=torch.cat([uncond_embeds, cond_embeds]),
                pooled_projections=torch.cat([uncond_pooled, cond_pooled]),
                timestep=t.expand(2),
            ).sample

            pred_uncond, pred_cond = pred.chunk(2)
            pred_guided = pred_uncond + 7.0 * (pred_cond - pred_uncond)

            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            latent = latent + (sigma_next - sigma) * pred_guided

            if i % 7 == 0:
                print(f"  Step {i}/{NUM_STEPS}, t={t.item():.1f}")

    # --- VAE decode ---
    scaling_factor = pipe.vae.config.scaling_factor   # 1.5305
    shift_factor = pipe.vae.config.shift_factor       # 0.0609
    latent_scaled = latent / scaling_factor + shift_factor

    with torch.no_grad():
        image = pipe.vae.decode(latent_scaled).sample

    np.savez(str(ASSETS / "vae_decoder.io.npz"),
        latent=latent_scaled.numpy(),
        image=image.numpy(),
    )
    print("Saved vae_decoder.io.npz")

    # Save reference image
    image_np = ((image[0].permute(1, 2, 0).clamp(-1, 1) + 1) / 2 * 255).byte().numpy()
    from PIL import Image
    Image.fromarray(image_np).save(str(ASSETS / "reference.png"))
    print(f"Saved reference image to {ASSETS / 'reference.png'}")


if __name__ == "__main__":
    main()
