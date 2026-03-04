use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::*;
use clap::Parser;
use float_ord::FloatOrd;
use itertools::Itertools;
use tract_rs::prelude::*;
use tract_rs::Nnef;

mod greedy;
mod beam;
mod alsd;

#[derive(Parser)]
#[command(about = "NeMo Parakeet ASR inference")]
struct Args {
    #[command(flatten)]
    beam: beam::BeamConfig,
    #[command(flatten)]
    alsd: alsd::AlsdConfig,
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
}

#[derive(Default)]
pub(crate) struct CallStats {
    calls: u32,
    total_batch: u64,
    total_us: u64,
}

impl CallStats {
    pub(crate) fn record(&mut self, batch: usize, elapsed: std::time::Duration) {
        self.calls += 1;
        self.total_batch += batch as u64;
        self.total_us += elapsed.as_micros() as u64;
    }

    pub(crate) fn total_ms(&self) -> f64 {
        self.total_us as f64 / 1000.0
    }
}

impl std::fmt::Debug for CallStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let avg_batch =
            if self.calls == 0 { 0.0 } else { self.total_batch as f64 / self.calls as f64 };
        let avg_ms =
            if self.calls == 0 { 0.0 } else { self.total_us as f64 / self.calls as f64 / 1000.0 };
        let total_ms = self.total_us as f64 / 1000.0;
        write!(
            f,
            "calls={:5}  avg_batch={avg_batch:.1}  avg={avg_ms:.3}ms  total={total_ms:.1}ms",
            self.calls
        )
    }
}

#[derive(Default)]
pub(crate) struct DecodingStats {
    pub(crate) preprocessor: CallStats,
    pub(crate) encoder: CallStats,
    pub(crate) decoder: CallStats,
    pub(crate) joint: CallStats,
}

impl DecodingStats {
    pub(crate) fn nn_ms(&self) -> f64 {
        self.preprocessor.total_ms()
            + self.encoder.total_ms()
            + self.decoder.total_ms()
            + self.joint.total_ms()
    }
}

pub(crate) fn argmax(slice: &[f32]) -> Option<usize> {
    slice.into_iter().position_max_by_key(|x| FloatOrd(**x))
}

pub(crate) fn log_softmax(xs: &[f32]) -> Vec<f32> {
    let max = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lse = xs.iter().map(|&x| (x - max).exp()).sum::<f32>().ln();
    xs.iter().map(|&x| x - max - lse).collect()
}

pub(crate) struct TdtModel {
    pub(crate) preprocessor: Runnable,
    pub(crate) encoder: Runnable,
    pub(crate) decoder: Runnable,
    pub(crate) joint: Runnable,
    pub(crate) vocab: Vec<String>,
    pub(crate) blank_id: usize,
}

impl TdtModel {
    fn load(model_dir: impl AsRef<Path>, nnef: &Nnef, gpu: &Runtime) -> Result<TdtModel> {
        let model_dir = model_dir.as_ref();
        let config: serde_json::Value =
            serde_json::from_reader(File::open(model_dir.join("model_config.json"))?)?;
        let blank_id =
            config.pointer("/decoder/vocab_size").unwrap().as_i64().unwrap() as usize;
        let vocab = config.pointer("/joint/vocabulary").unwrap().as_array().unwrap();
        let vocab: Vec<String> = vocab.iter().map(|v| v.as_str().unwrap().to_owned()).collect();

        let preprocessor =
            nnef.load(model_dir.join("preprocessor.nnef.tgz"))?.into_runnable()?;

        let mut encoder = nnef.load(model_dir.join("encoder.nnef.tgz"))?;
        encoder.transform("transformers-detect-all")?;
        let encoder = gpu.prepare(encoder)?;

        let decoder = nnef.load(model_dir.join("decoder.nnef.tgz"))?;
        let decoder = gpu.prepare(decoder)?;

        let joint = nnef.load(model_dir.join("joint.nnef.tgz"))?;
        let joint = gpu.prepare(joint)?;

        Ok(TdtModel { preprocessor, encoder, decoder, joint, vocab, blank_id })
    }
}

fn collect_wavs_from_dir(dir: &Path) -> Vec<PathBuf> {
    let mut results = Vec::new();
    if let Some(entries) = std::fs::read_dir(dir).ok() {
        for entry in entries.flatten() {
            let path: PathBuf = entry.path();
            if path.is_dir() {
                results.extend(collect_wavs_from_dir(&path));
            } else if path.extension().and_then(|e: &std::ffi::OsStr| e.to_str()) == Some("wav") {
                results.push(path);
            }
        }
    }
    results
}

fn collect_wavs(inputs: &[PathBuf]) -> Vec<PathBuf> {
    let mut results = Vec::new();
    for input in inputs {
        if input.is_dir() {
            results.extend(collect_wavs_from_dir(input));
        } else {
            results.push(input.clone());
        }
    }
    results.sort();
    results
}

fn load_wav(path: &Path) -> Result<(Vec<f32>, f64)> {
    let mut wav_reader = hound::WavReader::open(path)?;
    let sample_rate = wav_reader.spec().sample_rate as f64;
    let samples: Vec<f32> = wav_reader.samples::<i16>()
        .map(|x| x.unwrap() as f32)
        .collect();
    let audio_duration_s = samples.len() as f64 / sample_rate;
    Ok((samples, audio_duration_s))
}

fn clean(s: &str) -> String {
    s.replace('▁', " ").trim_start().to_owned()
}

fn print_result(
    label: &str,
    transcript: &str,
    reference: &str,
    elapsed_ms: f64,
    rtfx: f64,
    nn_ms: f64,
    stats: &DecodingStats,
) {
    let mark = if transcript == reference { "\x1b[32m✓\x1b[0m" } else { "\x1b[31m✗\x1b[0m" };
    eprintln!(
        "  [{label}] {elapsed_ms:.1}ms  RTFx={rtfx:.1}  nn={nn_ms:.1}ms  host={:.1}ms  {mark}",
        elapsed_ms - nn_ms
    );
    if transcript != reference {
        eprintln!("    ref: {}", clean(reference));
        eprintln!("    got: {}", clean(transcript));
    }
    eprintln!("    [preprocessor] {:?}", stats.preprocessor);
    eprintln!("    [encoder]      {:?}", stats.encoder);
    eprintln!("    [decoder]      {:?}", stats.decoder);
    eprintln!("    [joint]        {:?}", stats.joint);
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let nnef = tract_rs::nnef()?.with_tract_core()?.with_tract_transformers()?;
    let gpu = ["cuda", "metal", "default"]
        .iter()
        .find_map(|rt| tract_rs::runtime_for_name(rt).ok())
        .unwrap();

    let model = TdtModel::load(Path::new("assets/model"), &nnef, &gpu)?;

    let gt_cfg = beam::BeamConfig { beam_size: 10, dur_beam_k: 5 };
    let wavs = collect_wavs(&args.inputs);

    // Warmup on first file (all 4 decoders)
    if let Some(first) = wavs.first() {
        let (wav, _) = load_wav(first)?;
        greedy::transcribe_greedy(&model, &wav)?;
        beam::transcribe_beam(&model, &wav, &args.beam)?;
        alsd::transcribe_alsd(&model, &wav, &args.alsd)?;
        beam::transcribe_beam(&model, &wav, &gt_cfg)?;
    }

    for wav_path in &wavs {
        let (wav, audio_s) = load_wav(wav_path)?;

        // Reference — high-quality beam as ground truth (silent)
        let (reference, _) = beam::transcribe_beam(&model, &wav, &gt_cfg)?;
        eprintln!("{}  ({audio_s:.1}s)  {}", wav_path.display(), clean(&reference));

        // Greedy
        let t = Instant::now();
        let (transcript, stats) = greedy::transcribe_greedy(&model, &wav)?;
        let elapsed = t.elapsed().as_secs_f64();
        let elapsed_ms = elapsed * 1000.0;
        print_result("greedy", &transcript, &reference, elapsed_ms, audio_s / elapsed, stats.nn_ms(), &stats);

        // Beam
        let t = Instant::now();
        let (transcript, stats) = beam::transcribe_beam(&model, &wav, &args.beam)?;
        let elapsed = t.elapsed().as_secs_f64();
        let elapsed_ms = elapsed * 1000.0;
        print_result("beam", &transcript, &reference, elapsed_ms, audio_s / elapsed, stats.nn_ms(), &stats);

        // ALSD
        let t = Instant::now();
        let (transcript, stats) = alsd::transcribe_alsd(&model, &wav, &args.alsd)?;
        let elapsed = t.elapsed().as_secs_f64();
        let elapsed_ms = elapsed * 1000.0;
        print_result("alsd", &transcript, &reference, elapsed_ms, audio_s / elapsed, stats.nn_ms(), &stats);
    }

    Ok(())
}
