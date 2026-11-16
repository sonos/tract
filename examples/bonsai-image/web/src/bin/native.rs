//! The browser pipeline, run natively, to check its images:
//! `bonsai-native <models dir> <prompt> <out.png>`
//!
//! The models dir holds the three NNEF archives and `tokenizer.json`. `BONSAI_SIZE` (`HxW`),
//! `BONSAI_STEPS` and `BONSAI_SEED` set the rest.

/// The browser build has no use for it.
#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    native::main()
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::collections::HashMap;
    use std::fs;
    use std::path::Path;
    use std::time::Instant;

    use anyhow::{Context, Result, anyhow};
    use bonsai_web::pipeline::{Progress, generate, load};
    use tract::prelude::*;

    pub fn main() -> Result<()> {
        let args: Vec<String> = std::env::args().collect();
        let [_, models, prompt, out] = &args[..] else {
            anyhow::bail!("usage: bonsai-native <models dir> <prompt> <out.png>");
        };
        let models = Path::new(models);
        let tokenizer = tokenizers::Tokenizer::from_file(models.join("tokenizer.json"))
            .map_err(|e| anyhow!("{e}"))?;
        let env = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_string());
        let (height, width) = env("BONSAI_SIZE", "256x256")
            .split_once('x')
            .and_then(|(h, w)| Some((h.parse::<usize>().ok()?, w.parse::<usize>().ok()?)))
            .context("BONSAI_SIZE: HxW")?;
        let steps: usize = env("BONSAI_STEPS", "4").parse()?;
        let seed: u64 = env("BONSAI_SEED", "42").parse()?;

        let mut stages: HashMap<&'static str, Runnable> = HashMap::new();
        let mut stage = |name: &'static str| -> Result<Runnable> {
            if let Some(model) = stages.get(name) {
                return Ok(model.clone());
            }
            let file = match name {
                "text_encoder" => "text_encoder_gptq.nnef.tar".to_string(),
                _ => format!("{name}.nnef.tar"),
            };
            let t = Instant::now();
            let model = load(&mut fs::File::open(models.join(&file))?)
                .with_context(|| format!("loading {file}"))?;
            eprintln!("{name} ready in {:.2?}", t.elapsed());
            stages.insert(name, model.clone());
            Ok(model)
        };
        let t = Instant::now();
        let rgba = pollster::block_on(generate(
            &tokenizer,
            prompt,
            steps,
            seed,
            width,
            height,
            &mut stage,
            &mut |p| {
                if let Progress::Step(i) = p {
                    eprintln!("step {}/{steps}", i + 1);
                }
            },
            &mut None,
        ))?;
        eprintln!("{out} in {:.2?}", t.elapsed());
        let mut encoder = png::Encoder::new(fs::File::create(out)?, width as u32, height as u32);
        encoder.set_color(png::ColorType::Rgba);
        encoder.write_header()?.write_image_data(&rgba)?;
        Ok(())
    }
}
