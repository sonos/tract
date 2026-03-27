#!/usr/bin/env python3
"""Generate reference data for Stable Diffusion 1.5 validation."""

import torch
import numpy as np
from pathlib import Path
from diffusers import StableDiffusionPipeline

ASSETS = Path("assets")
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
PROMPT = "a photo of a cat"
SEED = 42
NUM_STEPS = 20
GUIDANCE_SCALE = 7.5

def main():
    print(f"Loading {MODEL_ID}...")
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    pipe = pipe.to("cpu")

    # Tokenize
    tokenizer = pipe.tokenizer
    tokens = tokenizer(
        PROMPT,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids  # (1, 77)
    print(f"Token IDs shape: {input_ids.shape}")

    # Unconditional tokens (empty prompt for classifier-free guidance)
    uncond_tokens = tokenizer(
        "",
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    uncond_input_ids = uncond_tokens.input_ids

    # Text encoder
    with torch.no_grad():
        text_output = pipe.text_encoder(input_ids)
        text_embeddings = text_output[0]  # last_hidden_state (1, 77, 768)
        uncond_output = pipe.text_encoder(uncond_input_ids)
        uncond_embeddings = uncond_output[0]

    # Concat for classifier-free guidance: [uncond, cond]
    text_embeddings_cfg = torch.cat([uncond_embeddings, text_embeddings])  # (2, 77, 768)
    print(f"Text embeddings shape: {text_embeddings_cfg.shape}")

    # Scheduler setup
    pipe.scheduler.set_timesteps(NUM_STEPS)
    timesteps = pipe.scheduler.timesteps
    print(f"Timesteps: {timesteps}")

    # Initial latent
    generator = torch.Generator().manual_seed(SEED)
    latent = torch.randn(1, 4, 64, 64, generator=generator)
    latent = latent * pipe.scheduler.init_noise_sigma
    print(f"Initial latent shape: {latent.shape}")

    # Denoising loop
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latent] * 2)  # (2, 4, 64, 64)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings_cfg,
            ).sample

            # Classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)

            # Scheduler step
            latent = pipe.scheduler.step(noise_pred, t, latent).prev_sample

            if i % 5 == 0:
                print(f"  Step {i}/{NUM_STEPS}, t={t.item()}")

    print(f"Final latent shape: {latent.shape}")

    # VAE decode
    with torch.no_grad():
        latent_scaled = latent / pipe.vae.config.scaling_factor
        image = pipe.vae.decode(latent_scaled).sample

    print(f"Image shape: {image.shape}")

    # --- Save per-model I/O for validation ---
    # Text encoder (has two outputs: last_hidden_state and pooler_output)
    with torch.no_grad():
        te_out = pipe.text_encoder(input_ids)
    np.savez(str(ASSETS / "text_encoder.io.npz"),
        input_ids=input_ids.numpy(),
        last_hidden_state=te_out[0].numpy(),
        pooler_output=te_out[1].numpy(),
    )
    print("Saved text_encoder.io.npz")

    # One UNet step (first step)
    pipe.scheduler.set_timesteps(NUM_STEPS)
    ts = pipe.scheduler.timesteps
    generator2 = torch.Generator().manual_seed(SEED)
    lat = torch.randn(1, 4, 64, 64, generator=generator2) * pipe.scheduler.init_noise_sigma
    t0 = ts[0]
    lat_in = pipe.scheduler.scale_model_input(lat, t0)
    with torch.no_grad():
        unet_out = pipe.unet(lat_in, t0, encoder_hidden_states=text_embeddings).sample
    np.savez(str(ASSETS / "unet.io.npz"),
        sample=lat_in.numpy(),
        timestep=np.array([t0.item()], dtype=np.int64),
        encoder_hidden_states=text_embeddings.numpy(),
        noise_pred=unet_out.numpy(),
    )
    print("Saved unet.io.npz")

    # VAE decoder
    np.savez(str(ASSETS / "vae_decoder.io.npz"),
        latent=latent_scaled.numpy(),
        image=image.numpy(),
    )
    print("Saved vae_decoder.io.npz")

    # --- Save data for Rust ---
    input_ids.numpy().tofile(str(ASSETS / "input_ids.bin"))
    uncond_input_ids.numpy().tofile(str(ASSETS / "uncond_input_ids.bin"))
    torch.randn(1, 4, 64, 64, generator=torch.Generator().manual_seed(SEED)).numpy().tofile(str(ASSETS / "initial_latent.bin"))
    timesteps.numpy().tofile(str(ASSETS / "timesteps.bin"))

    sigma = pipe.scheduler.init_noise_sigma
    sigma_val = sigma.item() if hasattr(sigma, 'item') else sigma
    with open(str(ASSETS / "params.txt"), "w") as f:
        f.write(f"vae_scaling_factor={pipe.vae.config.scaling_factor}\n")
        f.write(f"init_noise_sigma={sigma_val}\n")
        f.write(f"num_steps={NUM_STEPS}\n")

    print("Saved reference data")

    # Also save image as PNG for visual inspection
    image_np = ((image[0].permute(1, 2, 0).clamp(-1, 1) + 1) / 2 * 255).byte().numpy()
    from PIL import Image
    Image.fromarray(image_np).save(str(ASSETS / "reference.png"))
    print(f"Saved reference image to {ASSETS / 'reference.png'}")


if __name__ == "__main__":
    main()
