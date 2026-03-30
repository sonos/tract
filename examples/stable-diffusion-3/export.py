#!/usr/bin/env python3
"""Export Stable Diffusion 3 Medium components to ONNX."""

import torch
from pathlib import Path
from transformers import CLIPTokenizerFast

ASSETS = Path("assets")
MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-t5", action="store_true", help="Include T5-XXL encoder (~18GB)")
    args = parser.parse_args()

    from diffusers import StableDiffusion3Pipeline

    extra = {} if args.with_t5 else dict(text_encoder_3=None, tokenizer_3=None)
    print(f"Loading {MODEL_ID} ({'with' if args.with_t5 else 'without'} T5)...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, **extra,
    )
    pipe = pipe.to("cpu")

    ASSETS.mkdir(parents=True, exist_ok=True)

    # --- Text Encoder 1 (CLIP-L/14, hidden_size=768) ---
    print("Exporting text_encoder (CLIP-L)...")
    text_encoder = pipe.text_encoder
    text_encoder.eval()
    input_ids = torch.zeros(1, 77, dtype=torch.int64)
    with torch.no_grad():
        torch.onnx.export(
            text_encoder,
            (input_ids,),
            str(ASSETS / "text_encoder.onnx"),
            input_names=["input_ids"],
            output_names=["text_embeds", "last_hidden_state"],
            opset_version=18,
        )
    print(f"  Exported to {ASSETS / 'text_encoder.onnx'}")

    # --- Text Encoder 2 (OpenCLIP bigG/14, hidden_size=1280) ---
    print("Exporting text_encoder_2 (CLIP-G)...")
    text_encoder_2 = pipe.text_encoder_2
    text_encoder_2.eval()
    with torch.no_grad():
        torch.onnx.export(
            text_encoder_2,
            (input_ids,),
            str(ASSETS / "text_encoder_2.onnx"),
            input_names=["input_ids"],
            output_names=["text_embeds", "last_hidden_state"],
            opset_version=18,
        )
    print(f"  Exported to {ASSETS / 'text_encoder_2.onnx'}")

    # --- Text Encoder 3 (T5-XXL, hidden_size=4096, optional) ---
    if args.with_t5:
        print("Exporting text_encoder_3 (T5-XXL)...")
        text_encoder_3 = pipe.text_encoder_3
        text_encoder_3.eval()
        t5_input_ids = torch.zeros(1, 256, dtype=torch.int64)
        with torch.no_grad():
            torch.onnx.export(
                text_encoder_3,
                (t5_input_ids,),
                str(ASSETS / "text_encoder_3.onnx"),
                input_names=["input_ids"],
                output_names=["last_hidden_state"],
                opset_version=18,
            )
        print(f"  Exported to {ASSETS / 'text_encoder_3.onnx'}")

    # --- VAE Decoder ---
    print("Exporting vae_decoder...")
    vae = pipe.vae
    vae.eval()

    class VaeDecoder(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latent):
            return self.vae.decode(latent).sample

    vae_decoder = VaeDecoder(vae)
    latent = torch.randn(1, 16, 128, 128)
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

    # --- Transformer (MMDiT) ---
    seq_len = 333 if args.with_t5 else 77
    print(f"Exporting transformer (MMDiT, seq={seq_len})...")
    transformer = pipe.transformer
    transformer.eval()

    hidden_states = torch.randn(2, 16, 128, 128)
    encoder_hidden_states = torch.randn(2, seq_len, 4096)
    pooled_projections = torch.randn(2, 2048)
    timestep = torch.tensor([500.0, 500.0])

    class TransformerWrapper(torch.nn.Module):
        def __init__(self, transformer):
            super().__init__()
            self.transformer = transformer

        def forward(self, hidden_states, encoder_hidden_states, pooled_projections, timestep):
            return self.transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,
            ).sample

    wrapper = TransformerWrapper(transformer)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (hidden_states, encoder_hidden_states, pooled_projections, timestep),
            str(ASSETS / "transformer.onnx"),
            input_names=["hidden_states", "encoder_hidden_states", "pooled_projections", "timestep"],
            output_names=["output"],
            opset_version=18,
            dynamic_axes={
                "hidden_states": {0: "batch"},
                "encoder_hidden_states": {0: "batch"},
                "pooled_projections": {0: "batch"},
                "timestep": {0: "batch"},
                "output": {0: "batch"},
            },
        )
    print(f"  Exported to {ASSETS / 'transformer.onnx'}")

    # Save tokenizers
    print("Saving tokenizers...")
    tok = CLIPTokenizerFast.from_pretrained(MODEL_ID, subfolder="tokenizer")
    tok.save_pretrained(str(ASSETS / "tokenizer"))
    if args.with_t5:
        from transformers import AutoTokenizer
        tok3 = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer_3")
        tok3.save_pretrained(str(ASSETS / "tokenizer_3"))

    print("Export complete.")


if __name__ == "__main__":
    main()
