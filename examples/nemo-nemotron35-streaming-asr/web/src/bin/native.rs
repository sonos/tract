//! The browser pipeline, run natively on a WebGPU device (Vulkan, Metal or DX12), to check it:
//! `nemotron-native <graphs dir> <hf snapshot dir> <clip.wav> [lang_id]`
//!
//! The graphs dir holds what `export.sh` writes; the snapshot dir, `model.safetensors` and
//! `tokenizer.json`. The clip is mono 16-bit, at 16 kHz or a multiple of it, and is fed in the
//! page's 20 ms blocks. `lang_id` defaults to automatic language detection.

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

    use anyhow::{Result, bail};
    use nemotron_web::{AUTO_LANGUAGE, Decimator, Graphs, Models, Transcriber};

    pub fn main() -> Result<()> {
        let args: Vec<String> = std::env::args().collect();
        let (graphs, hf, wav, lang_id) = match &args[1..] {
            [g, h, w] => (g, h, w, AUTO_LANGUAGE),
            [g, h, w, l] => (g, h, w, l.parse()?),
            _ => {
                bail!("usage: nemotron-native <graphs dir> <hf snapshot dir> <clip.wav> [lang_id]")
            }
        };
        let (graphs, hf) = (Path::new(graphs), Path::new(hf));
        let graph = |name: &str| fs::read(graphs.join(format!("{name}.nnef.tar")));
        let (preprocessor, encoder, decoder, joint) =
            (graph("preprocessor")?, graph("encoder")?, graph("decoder")?, graph("joint")?);
        let t = Instant::now();
        let models = Models::load(
            Graphs {
                preprocessor: &preprocessor,
                encoder: &encoder,
                decoder: &decoder,
                joint: &joint,
            },
            &fs::read_to_string(hf.join("tokenizer.json"))?,
            || Ok(fs::File::open(hf.join("model.safetensors"))?),
        )?;
        eprintln!("ready in {:.2?}", t.elapsed());

        let mut reader = hound::WavReader::open(wav)?;
        let rate = reader.spec().sample_rate as usize;
        let samples: Vec<f32> =
            reader.samples::<i16>().map(|s| Ok(s? as f32 / 32768.0)).collect::<Result<_>>()?;
        let mut decimator = Decimator::new(rate)?;
        let mut transcriber = Transcriber::new(&models, lang_id)?;
        let t = Instant::now();
        pollster::block_on(async {
            for block in samples.chunks(rate / 50) {
                transcriber.push(&decimator.push(block)).await?;
            }
            transcriber.finish().await
        })?;
        let audio = samples.len() as f32 / rate as f32;
        eprintln!("{audio:.1} s of audio in {:.2?}", t.elapsed());
        println!("{}", transcriber.text().trim());
        Ok(())
    }
}
