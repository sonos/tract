use std::time::Instant;

use anyhow::*;
use itertools::Itertools;
use ndarray::prelude::*;
use qwen3_asr::*;

/// Whole-utterance transcription: one pass over every 8 s attention window, one decode.
fn transcribe(models: &Models, signal: &[f32]) -> Result<String> {
    let padded = reflect(signal, true)
        .chain(signal.iter().copied())
        .chain(reflect(signal, false))
        .collect_vec();
    let log_mel = models.log_mel(&padded)?;
    let log_mel = log_mel.slice(s![.., ..signal.len() / HOP]);
    let max = log_mel.fold(f32::MIN, |m, &x| m.max(x));
    let frames = log_mel.shape()[1];
    let windows = (0..frames)
        .step_by(WINDOW_FRAMES)
        .map(|w| models.encode_window(log_mel.slice(s![.., w..frames.min(w + WINDOW_FRAMES)]), max))
        .collect::<Result<Vec<_>>>()?;
    let audio = ndarray::concatenate(Axis(0), &windows.iter().map(|w| w.view()).collect_vec())?;
    let mut ids = models.prompt_head.clone();
    ids.extend(models.audio_prompt(audio.shape()[0], &[]));
    models.decode(&models.generate(&ids, audio.view(), &models.empty_kv()?)?)
}

fn main() -> anyhow::Result<()> {
    let wav = std::env::args().nth(1);
    let models = Models::load()?;
    let signal = load_wav(wav.as_deref().unwrap_or(SAMPLE_WAV))?;
    let start = Instant::now();
    let transcript = transcribe(&models, &signal)?;
    eprintln!("{:.2}s of audio in {:?}", signal.len() as f32 / SAMPLE_RATE as f32, start.elapsed());
    println!("{transcript}");
    if wav.is_none() {
        assert_eq!(transcript, SAMPLE_TRANSCRIPT);
    }
    Ok(())
}
