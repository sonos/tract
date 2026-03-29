#!/usr/bin/env python3
"""Generate reference I/O data for SDXL model validation."""

import torch
import numpy as np
from pathlib import Path
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

ASSETS = Path("assets")
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
PROMPT = "a photo of a cat"
SEED = 42
NUM_STEPS = 20

def main():
    print(f"Loading {MODEL_ID}...")
    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float32, variant="fp16")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
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

    print(f"Token IDs: {input_ids.shape}, Token IDs 2: {input_ids_2.shape}")

    # --- Text Encoder 1 ---
    with torch.no_grad():
        te1_cond = pipe.text_encoder(input_ids)
        te1_uncond = pipe.text_encoder(uncond_input_ids)

    np.savez(str(ASSETS / "text_encoder.io.npz"),
        input_ids=input_ids.numpy(),
        last_hidden_state=te1_cond.last_hidden_state.numpy(),
        pooler_output=te1_cond.pooler_output.numpy(),
    )
    print("Saved text_encoder.io.npz")

    # --- Text Encoder 2 ---
    with torch.no_grad():
        te2_cond = pipe.text_encoder_2(input_ids_2)
        te2_uncond = pipe.text_encoder_2(uncond_input_ids_2)

    np.savez(str(ASSETS / "text_encoder_2.io.npz"),
        input_ids=input_ids_2.numpy(),
        last_hidden_state=te2_cond.last_hidden_state.numpy(),
        text_embeds=te2_cond.text_embeds.numpy(),
    )
    print("Saved text_encoder_2.io.npz")

    # --- Concat text embeddings ---
    cond_hidden = torch.cat([te1_cond.last_hidden_state, te2_cond.last_hidden_state], dim=-1)  # (1,77,2048)
    uncond_hidden = torch.cat([te1_uncond.last_hidden_state, te2_uncond.last_hidden_state], dim=-1)
    text_embeddings = cond_hidden  # for single-image reference
    print(f"Text embeddings: {text_embeddings.shape}")

    # Pooled embeddings from text_encoder_2
    cond_pooled = te2_cond.text_embeds  # (1, 1280)
    uncond_pooled = te2_uncond.text_embeds

    # time_ids: [original_h, original_w, crop_top, crop_left, target_h, target_w]
    time_ids = torch.tensor([[1024., 1024., 0., 0., 1024., 1024.]])

    # --- UNet (one step) ---
    pipe.scheduler.set_timesteps(NUM_STEPS)
    ts = pipe.scheduler.timesteps
    generator = torch.Generator().manual_seed(SEED)
    latent = torch.randn(1, 4, 128, 128, generator=generator) * pipe.scheduler.init_noise_sigma
    t0 = ts[0]
    lat_in = pipe.scheduler.scale_model_input(latent, t0)

    with torch.no_grad():
        unet_out = pipe.unet(
            lat_in, t0,
            encoder_hidden_states=text_embeddings,
            added_cond_kwargs={"text_embeds": cond_pooled, "time_ids": time_ids},
        ).sample

    np.savez(str(ASSETS / "unet.io.npz"),
        sample=lat_in.numpy(),
        timestep=np.array([t0.item()], dtype=np.int64),
        encoder_hidden_states=text_embeddings.numpy(),
        time_ids=time_ids.numpy(),
        text_embeds=cond_pooled.numpy(),
        noise_pred=unet_out.numpy(),
    )
    print("Saved unet.io.npz")

    # --- Full denoising loop ---
    print(f"Running full pipeline ({NUM_STEPS} steps)...")
    latent = torch.randn(1, 4, 128, 128, generator=torch.Generator().manual_seed(SEED)) * pipe.scheduler.init_noise_sigma
    with torch.no_grad():
        for i, t in enumerate(ts):
            latent_model_input = torch.cat([latent] * 2)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = pipe.unet(
                latent_model_input, t,
                encoder_hidden_states=torch.cat([uncond_hidden, cond_hidden]),
                added_cond_kwargs={
                    "text_embeds": torch.cat([uncond_pooled, cond_pooled]),
                    "time_ids": torch.cat([time_ids, time_ids]),
                },
            ).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
            latent = pipe.scheduler.step(noise_pred, t, latent).prev_sample
            if i % 5 == 0:
                print(f"  Step {i}/{NUM_STEPS}, t={t.item()}")

    # --- VAE decode ---
    with torch.no_grad():
        latent_scaled = latent / pipe.vae.config.scaling_factor
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
