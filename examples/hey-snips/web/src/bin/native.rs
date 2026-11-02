//! Streams WAV files through the detector the page uses, in the page's 20 ms blocks, and prints each
//! file's peak score, when it peaks, and how much faster than real time it ran.
//!
//! Usage: hey-snips-native <hey_snips.nnef.tar> <clip.wav>...
//! Clips are 16-bit PCM, mono, at 48 kHz; each is fed after and before a second of silence.

use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use hey_snips_web::Detector;
use hey_snips_web::SAMPLE_RATE;

fn read_wav(path: &str) -> Result<Vec<f32>> {
    let data = std::fs::read(path)?;
    ensure!(data.len() >= 12 && &data[..4] == b"RIFF" && &data[8..12] == b"WAVE", "Not a WAV file");
    let mut rest = &data[12..];
    let mut format = false;
    while rest.len() >= 8 {
        let (id, len) = (&rest[..4], u32::from_le_bytes(rest[4..8].try_into()?) as usize);
        let body = rest.get(8..8 + len).context("Truncated WAV chunk")?;
        match id {
            b"fmt " => {
                let channels = u16::from_le_bytes(body[2..4].try_into()?);
                let bits = u16::from_le_bytes(body[14..16].try_into()?);
                let rate = u32::from_le_bytes(body[4..8].try_into()?);
                ensure!(channels == 1 && bits == 16, "Expected 16-bit mono PCM");
                ensure!(rate == SAMPLE_RATE, "Expected {SAMPLE_RATE} Hz, got {rate} Hz");
                format = true;
            }
            b"data" => {
                ensure!(format, "WAV data before its format");
                let samples = body.as_chunks::<2>().0.iter();
                return Ok(samples.map(|s| i16::from_le_bytes(*s) as f32 / 32768.0).collect());
            }
            _ => (),
        }
        rest = &rest[(8 + len + len % 2).min(rest.len())..];
    }
    bail!("No WAV data")
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = std::fs::read(args.next().context("Pass the model path")?)?;
    for path in args {
        let clip = read_wav(&path).with_context(|| format!("Reading {path}"))?;
        let silence = vec![0f32; SAMPLE_RATE as usize];
        let audio = [&silence[..], &clip, &silence].concat();
        let mut detector = Detector::new(&model)?;
        let start = Instant::now();
        let mut scores = vec![];
        for chunk in audio.chunks(SAMPLE_RATE as usize / 50) {
            scores.extend(detector.push(chunk)?);
        }
        let elapsed = start.elapsed().as_secs_f64();
        let (peak_frame, peak) = scores
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .context("Clip too short for a score")?;
        let window_end = (peak_frame + 182) as f64 * 0.01;
        println!(
            "{path}: peak {peak:6.2} for the window ending at {window_end:.2} s, {:.0}x real time",
            audio.len() as f64 / SAMPLE_RATE as f64 / elapsed
        );
    }
    Ok(())
}
