#!/usr/bin/env python3
"""Fits the linear map from DiT tokens to RGB that previews an image while it denoises.

Encodes images through the Flux2 VAE into normalized packed latents, the space the DiT works in,
and solves a ridge regression from each token to its 16x16 pixel block. Writes src/latent_rgb.rs.

Usage: fit_preview.py <image>...
"""

import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLFlux2
from PIL import Image

MODEL_ID = "prism-ml/bonsai-image-ternary-4B-unpacked"
RIDGE = 1.0


def tokens_and_pixels(vae, path):
    image = Image.open(path).convert("RGB")
    rgb = np.asarray(image, dtype=np.float32) / 127.5 - 1
    x = torch.from_numpy(rgb).permute(2, 0, 1)[None]
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mode()
        z = z.reshape(1, 32, z.shape[2] // 2, 2, z.shape[3] // 2, 2).permute(0, 1, 3, 5, 2, 4)
        z = z.reshape(1, 128, z.shape[4], z.shape[5])
        mean = vae.bn.running_mean.view(1, -1, 1, 1)
        std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
        tokens = ((z - mean) / std)[0].permute(1, 2, 0).reshape(-1, 128).numpy()
    h, w = z.shape[2:]
    pixels = torch.nn.functional.avg_pool2d(x, 16)[0].permute(1, 2, 0).reshape(h * w, 3).numpy()
    return tokens, pixels


def main():
    images = sys.argv[1:]
    vae = AutoencoderKLFlux2.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32).eval()
    pairs = [tokens_and_pixels(vae, p) for p in images]
    held = len(images) // 5 or 1
    fit = (np.concatenate([t for t, _ in pairs[:-held]]), np.concatenate([p for _, p in pairs[:-held]]))
    test = (np.concatenate([t for t, _ in pairs[-held:]]), np.concatenate([p for _, p in pairs[-held:]]))
    design = lambda t: np.hstack([t, np.ones((len(t), 1))])
    a, b = design(fit[0]), fit[1]
    m = np.linalg.solve(a.T @ a + RIDGE * np.eye(129), a.T @ b)
    print(f"{len(a)} tokens, held-out mean abs error {np.abs(design(test[0]) @ m - test[1]).mean() * 127.5:.1f} / 255")
    rows = ",\n".join("    [" + ", ".join(f"{v:.5f}" for v in row) + "]" for row in m.T)
    Path(__file__).with_name("src").joinpath("latent_rgb.rs").write_text(
        "//! Linear map from a DiT token (and a constant) to RGB in [-1, 1], from `fit_preview.py`.\n\n"
        f"pub const LATENT_RGB: [[f32; 129]; 3] = [\n{rows},\n];\n"
    )


if __name__ == "__main__":
    main()
