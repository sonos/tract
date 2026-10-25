//! The browser pipeline, run natively, to check its images:
//! `sd15-native <graphs dir> <hf snapshot dir> <prompt> <out.png>`
//!
//! The graphs dir holds `text_encoder.nnef.tar`, `unet.nnef.tar` and `vae_decoder.nnef.tar`; the
//! weights are read from the snapshot's fp16 safetensors files. `SD15_NEG`, `SD15_STEPS`,
//! `SD15_SEEDS` (comma separated), `SD15_GUIDANCE` and `SD15_SAMPLER` (`dpm++2m-karras` or
//! `euler`) set the rest. A prompt of `@file` renders every line of the file, for every seed,
//! into `out` as a directory.

/// The browser build has no use for it.
#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    native::main()
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::fs;
    use std::path::Path;
    use std::time::Instant;

    use anyhow::{Context, Result};
    use sd15_web::pipeline::{Pipeline, Progress, Request, SIZE, load};
    use sd15_web::scheduler::{Sampler, Scheduler};
    use sd15_web::tokenizer::Tokenizer;

    pub fn main() -> Result<()> {
        let args: Vec<String> = std::env::args().collect();
        if args.get(1).map(String::as_str) == Some("--schedule") {
            let sampler = Sampler::parse(&args[2]).context("sampler")?;
            let s = Scheduler::new(sampler, args[3].parse()?);
            println!("{:?}\n{:?}", s.timesteps, s.sigmas);
            return Ok(());
        }
        let [_, graphs, hf, prompt, out] = &args[..] else {
            anyhow::bail!("usage: sd15-native <graphs dir> <hf snapshot dir> <prompt> <out.png>");
        };
        let (graphs, hf) = (Path::new(graphs), Path::new(hf));
        let tokenizer = Tokenizer::new(
            &fs::read_to_string(hf.join("tokenizer/vocab.json"))?,
            &fs::read_to_string(hf.join("tokenizer/merges.txt"))?,
        )?;
        let component = |name: &str, weights: &str| {
            let t = Instant::now();
            let graph = fs::read(graphs.join(format!("{name}.nnef.tar")))?;
            let model = load(&graph, fs::File::open(hf.join(weights))?)
                .with_context(|| format!("loading {name}"))?;
            eprintln!("{name} ready in {:.2?}", t.elapsed());
            anyhow::Ok(model)
        };
        let pipeline = Pipeline {
            tokenizer,
            text_encoder: component("text_encoder", "text_encoder/model.fp16.safetensors")?,
            unet: component("unet", "unet/diffusion_pytorch_model.fp16.safetensors")?,
            vae: component("vae_decoder", "vae/diffusion_pytorch_model.fp16.safetensors")?,
        };

        let env = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_string());
        let negative = env("SD15_NEG", "");
        let steps: usize = env("SD15_STEPS", "15").parse()?;
        let guidance: f32 = env("SD15_GUIDANCE", "7.5").parse()?;
        let sampler = Sampler::parse(&env("SD15_SAMPLER", "dpm++2m-karras"))
            .context("SD15_SAMPLER: euler or dpm++2m-karras")?;
        let seeds: Vec<u64> =
            env("SD15_SEEDS", "7").split(',').map(|s| s.parse()).collect::<Result<_, _>>()?;
        let jobs: Vec<(String, u64, String)> = match prompt.strip_prefix('@') {
            Some(file) => fs::read_to_string(file)?
                .lines()
                .filter(|l| !l.trim().is_empty())
                .enumerate()
                .flat_map(|(i, l)| {
                    seeds
                        .iter()
                        .map(move |s| (l.to_string(), *s, format!("{out}/p{i:02}-s{s}.png")))
                })
                .collect(),
            None => vec![(prompt.to_string(), seeds[0], out.to_string())],
        };
        for (prompt, seed, path) in &jobs {
            let request =
                Request { prompt, negative: &negative, steps, seed: *seed, guidance, sampler };
            let t = Instant::now();
            let rgba = pollster::block_on(pipeline.generate(&request, &mut |p| {
                if let Progress::Step(i, n) = p {
                    eprint!("\rstep {}/{n}", i + 1);
                }
            }))?;
            eprintln!("\n{path} in {:.2?}", t.elapsed());
            let mut encoder = png::Encoder::new(fs::File::create(path)?, SIZE as u32, SIZE as u32);
            encoder.set_color(png::ColorType::Rgba);
            encoder.write_header()?.write_image_data(&rgba)?;
        }
        Ok(())
    }
}
