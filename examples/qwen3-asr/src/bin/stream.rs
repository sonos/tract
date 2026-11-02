use std::time::Instant;

use anyhow::*;
use itertools::Itertools;
use ndarray::prelude::*;
use qwen3_asr::*;
use tract::prelude::*;

const STEP_SECS: f32 = 2.0;
const UNFIXED_STEPS: usize = 2;
const UNFIXED_TOKENS: usize = 5;

/// Incremental transcription. Finished 8 s windows are encoded once and their decoder KV kept;
/// each step re-encodes the current window and re-decodes everything after it, prefixed by the
/// previous output minus its last UNFIXED_TOKENS.
struct Stream<'m> {
    models: &'m Models,
    samples: Vec<f32>,
    log_mel: Array2<f32>,
    max: f32,
    frozen_windows: usize,
    frozen_kv: Vec<Tensor>,
    steps: usize,
    output: Vec<i64>,
}

impl<'m> Stream<'m> {
    fn new(models: &'m Models) -> Result<Stream<'m>> {
        let no_audio = Array2::zeros((0, HIDDEN));
        let (_, frozen_kv) =
            models.forward(&models.prompt_head, no_audio.view(), &models.empty_kv()?)?;
        Ok(Stream {
            models,
            samples: vec![],
            log_mel: Array2::zeros((MEL_BINS, 0)),
            max: f32::MIN,
            frozen_windows: 0,
            frozen_kv,
            steps: 0,
            output: vec![],
        })
    }

    fn push(&mut self, samples: &[f32]) -> Result<()> {
        self.samples.extend(samples);
        self.extend_log_mel(&[])
    }

    /// Computes the mel frames whose window has fully arrived. Whisper's max - 8 floor uses the
    /// running max, so it only departs from the offline features for bins over 80 dB below
    /// a later peak.
    fn extend_log_mel(&mut self, tail: &[f32]) -> Result<()> {
        if self.samples.len() <= N_FFT / 2 {
            return Ok(());
        }
        let done = self.log_mel.shape()[1];
        let padded = reflect(&self.samples, true)
            .chain(self.samples.iter().copied())
            .chain(tail.iter().copied())
            .skip(done * HOP)
            .collect_vec();
        let new = self.models.log_mel(&padded)?;
        self.max = new.fold(self.max, |m, &x| m.max(x));
        self.log_mel.append(Axis(1), new.view())?;
        Ok(())
    }

    fn step(&mut self) -> Result<String> {
        let frames = self.log_mel.shape()[1];
        while (self.frozen_windows + 1) * WINDOW_FRAMES <= frames {
            let start = self.frozen_windows * WINDOW_FRAMES;
            let window = self.log_mel.slice(s![.., start..start + WINDOW_FRAMES]);
            let audio = self.models.encode_window(window, self.max)?;
            let ids = vec![AUDIO_PAD; audio.shape()[0]];
            (_, self.frozen_kv) = self.models.forward(&ids, audio.view(), &self.frozen_kv)?;
            self.frozen_windows += 1;
        }
        let start = self.frozen_windows * WINDOW_FRAMES;
        let audio = if start < frames {
            self.models.encode_window(self.log_mel.slice(s![.., start..]), self.max)?
        } else {
            Array2::zeros((0, HIDDEN))
        };
        let prefix = if self.steps < UNFIXED_STEPS {
            &[][..]
        } else {
            &self.output[..self.output.len().saturating_sub(UNFIXED_TOKENS)]
        };
        let ids = self.models.audio_prompt(audio.shape()[0], prefix);
        let generated = self.models.generate(&ids, audio.view(), &self.frozen_kv)?;
        self.output = prefix.iter().copied().chain(generated).collect();
        self.steps += 1;
        self.models.decode(&self.output)
    }

    fn finish(&mut self) -> Result<String> {
        let tail = reflect(&self.samples, false).collect_vec();
        self.extend_log_mel(&tail)?;
        self.log_mel.slice_collapse(s![.., ..self.samples.len() / HOP]);
        self.step()
    }
}

fn main() -> anyhow::Result<()> {
    let wav = std::env::args().nth(1);
    let models = Models::load()?;
    let signal = load_wav(wav.as_deref().unwrap_or(SAMPLE_WAV))?;
    let mut stream = Stream::new(&models)?;
    let step = (STEP_SECS * SAMPLE_RATE as f32) as usize;
    let mut text = String::new();
    for chunk in signal.chunks(step) {
        stream.push(chunk)?;
        let start = Instant::now();
        text = if chunk.len() == step { stream.step()? } else { stream.finish()? };
        eprintln!(
            "{:6.2}s +{:4}ms {text}",
            stream.samples.len() as f32 / SAMPLE_RATE as f32,
            start.elapsed().as_millis()
        );
    }
    if signal.len().is_multiple_of(step) {
        text = stream.finish()?;
    }
    println!("{text}");
    if wav.is_none() {
        assert_eq!(text, SAMPLE_TRANSCRIPT);
    }
    Ok(())
}
