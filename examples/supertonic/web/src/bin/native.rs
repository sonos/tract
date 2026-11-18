//! The browser pipeline, run natively, to check its audio:
//! `supertonic-native <model dir> <voice> <lang> <text> <out.wav>`
//!
//! The model dir is a snapshot of a Supertonic repository on Hugging Face, with `onnx/` and
//! `voice_styles/`. `SUPERTONIC_RUNTIME` (`cpu` or `wgpu`), `SUPERTONIC_STEPS`,
//! `SUPERTONIC_SPEED` and `SUPERTONIC_SEED` set the rest.

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

    use anyhow::Result;
    use supertonic_web::engine::Graphs;
    use supertonic_web::engine::Request;
    use supertonic_web::engine::Tts;
    use supertonic_web::engine::Voice;
    use supertonic_web::text;

    fn env<T: std::str::FromStr>(name: &str, default: T) -> T {
        std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
    }

    pub fn main() -> Result<()> {
        let args: Vec<String> = std::env::args().collect();
        let [_, dir, voice, lang, input, out] = &args[..] else {
            anyhow::bail!("usage: supertonic-native <model dir> <voice> <lang> <text> <out.wav>");
        };
        let dir = Path::new(dir);
        let runtime: String = env("SUPERTONIC_RUNTIME", "cpu".to_string());
        let read = |name: &str| fs::read(dir.join("onnx").join(name));
        let graphs = Graphs {
            duration_predictor: read("duration_predictor.onnx")?,
            text_encoder: read("text_encoder.onnx")?,
            vector_estimator: read("vector_estimator.onnx")?,
            vocoder: read("vocoder.onnx")?,
            tts_json: read("tts.json")?,
            unicode_indexer_json: read("unicode_indexer.json")?,
        };
        let start = Instant::now();
        let tts = pollster::block_on(Tts::create(&runtime, &graphs))?;
        eprintln!("prepared on {runtime} in {:.2} s", start.elapsed().as_secs_f32());
        let voice =
            Voice::from_json(&fs::read(dir.join("voice_styles").join(format!("{voice}.json")))?)?;
        let steps = env("SUPERTONIC_STEPS", 8);
        let speed = env("SUPERTONIC_SPEED", 1.05);
        let seed = env("SUPERTONIC_SEED", 0u64);
        let silence = vec![0f32; tts.sample_rate() * 3 / 10];
        let mut wav = vec![];
        for (ix, chunk) in text::chunks(input, lang).iter().enumerate() {
            let start = Instant::now();
            let request = Request { text: chunk, lang, steps, speed, seed: seed + ix as u64 };
            let samples = pollster::block_on(tts.speak(&voice, &request))?;
            let seconds = samples.len() as f32 / tts.sample_rate() as f32;
            let took = start.elapsed().as_secs_f32();
            eprintln!(
                "chunk {ix}: {seconds:.2} s of audio in {took:.3} s ({:.1}x real time)",
                seconds / took
            );
            if ix > 0 {
                wav.extend_from_slice(&silence);
            }
            wav.extend(samples);
        }
        write_wav(Path::new(out), &wav, tts.sample_rate())
    }

    fn write_wav(path: &Path, samples: &[f32], rate: usize) -> Result<()> {
        let mut bytes = Vec::with_capacity(44 + samples.len() * 2);
        let data_len = (samples.len() * 2) as u32;
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&(rate as u32).to_le_bytes());
        bytes.extend_from_slice(&(rate as u32 * 2).to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_len.to_le_bytes());
        for s in samples {
            bytes.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
        }
        Ok(fs::write(path, bytes)?)
    }
}
