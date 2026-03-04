use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::*;
use clap::Parser;
use progress_bar::*;
use float_ord::FloatOrd;
use itertools::Itertools;
use tract_rs::prelude::*;
use tract_rs::Nnef;

mod greedy;
mod beam;
mod alsd;

#[derive(clap::ValueEnum, Clone)]
enum Decoder { Greedy, Beam, Alsd }

#[derive(Parser)]
#[command(about = "NeMo Parakeet ASR inference")]
struct Args {
    #[command(flatten)]
    beam: beam::BeamConfig,
    #[command(flatten)]
    alsd: alsd::AlsdConfig,
    #[arg(long, value_enum, default_value_t = Decoder::Greedy)]
    decoder: Decoder,
    #[arg(long)]
    stats: bool,
    #[arg(long)]
    no_details: bool,
    /// Run ground-truth decoder and write transcript to a .txt file beside each wav
    #[arg(long)]
    write_gt: bool,
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

fn decoder_label(args: &Args) -> String {
    match args.decoder {
        Decoder::Greedy => "greedy".to_string(),
        Decoder::Beam => format!("beam  beam_size={}  beam_dur_k={}", args.beam.beam_size, args.beam.beam_dur_k),
        Decoder::Alsd => format!("alsd  beam_size={}  beam_dur_k={}  max_symbols={}", args.alsd.alsd_beam_size, args.alsd.alsd_beam_dur_k, args.alsd.alsd_max_symbols_per_frame),
    }
}

fn run_decoder(model: &TdtModel, wav: &[f32], args: &Args) -> Result<(String, DecodingStats)> {
    match args.decoder {
        Decoder::Greedy => greedy::transcribe_greedy(model, wav),
        Decoder::Beam   => beam::transcribe_beam(model, wav, &args.beam),
        Decoder::Alsd   => alsd::transcribe_alsd(model, wav, &args.alsd),
    }
}

fn write_gt(model: &TdtModel, wavs: &[PathBuf], no_details: bool) -> anyhow::Result<()> {
    let gt_cfg = beam::BeamConfig { beam_size: 10, beam_dur_k: 5 };
    // Warmup
    if let Some(first) = wavs.first() {
        let (wav, _) = load_wav(first)?;
        beam::transcribe_beam(model, &wav, &gt_cfg)?;
    }
    if no_details {
        init_progress_bar_with_eta(wavs.len());
        set_progress_bar_action("Writing GT", Color::Blue, Style::Bold);
    }
    for wav_path in wavs {
        let (wav, _) = load_wav(wav_path)?;
        let (transcript, _) = beam::transcribe_beam(model, &wav, &gt_cfg)?;
        let txt_path = wav_path.with_extension("txt");
        std::fs::write(&txt_path, clean(&transcript))?;
        if no_details {
            inc_progress_bar();
        } else {
            eprintln!("{} -> {}", wav_path.display(), txt_path.display());
        }
    }
    if no_details {
        finalize_progress_bar();
    }
    eprintln!("wrote {} transcript(s)", wavs.len());
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let nnef = tract_rs::nnef()?.with_tract_core()?.with_tract_transformers()?;
    let gpu = ["cuda", "metal", "default"]
        .iter()
        .find_map(|rt| tract_rs::runtime_for_name(rt).ok())
        .unwrap();

    let model = TdtModel::load(Path::new("assets/model"), &nnef, &gpu)?;
    let wavs = collect_wavs(&args.inputs);

    if args.write_gt {
        return write_gt(&model, &wavs, args.no_details);
    }

    let label = decoder_label(&args);
    if !args.no_details {
        eprintln!("{}  files={}", label, wavs.len());
    } else {
        init_progress_bar_with_eta(wavs.len());
        set_progress_bar_action("Decoding", Color::Blue, Style::Bold);
    }

    // Warmup on first file
    if let Some(first) = wavs.first() {
        let (wav, _) = load_wav(first)?;
        run_decoder(&model, &wav, &args)?;
    }

    let mut total_audio_s = 0.0f64;
    let mut total_elapsed_s = 0.0f64;
    let mut exact = 0usize;

    for wav_path in &wavs {
        let (wav, audio_s) = load_wav(wav_path)?;
        let reference = std::fs::read_to_string(wav_path.with_extension("txt"))
            .with_context(|| format!("no ground-truth transcript for {} (run with --write-gt first)", wav_path.display()))?;

        let t = Instant::now();
        let (transcript, stats) = run_decoder(&model, &wav, &args)?;
        let elapsed = t.elapsed().as_secs_f64();

        total_audio_s += audio_s;
        total_elapsed_s += elapsed;

        let ok = transcript == reference;
        if ok { exact += 1; }

        if args.no_details {
            inc_progress_bar();
        } else {
            let mark = if ok { "\x1b[32m✓\x1b[0m" } else { "\x1b[31m✗\x1b[0m" };
            let elapsed_ms = elapsed * 1000.0;
            let nn_ms = stats.nn_ms();
            eprintln!("{}  {mark}  {audio_s:.1}s  {elapsed_ms:.1}ms  RTFx={:.1}  {}",
                      wav_path.display(), audio_s / elapsed, clean(&transcript));
            if !ok {
                eprintln!("  ref: {}", clean(&reference));
                eprintln!("  got: {}", clean(&transcript));
            }
            if args.stats {
                eprintln!("  {elapsed_ms:.1}ms  RTFx={:.1}  nn={nn_ms:.1}ms  host={:.1}ms",
                          audio_s / elapsed, elapsed_ms - nn_ms);
                eprintln!("  [preprocessor] {:?}", stats.preprocessor);
                eprintln!("  [encoder]      {:?}", stats.encoder);
                eprintln!("  [decoder]      {:?}", stats.decoder);
                eprintln!("  [joint]        {:?}", stats.joint);
            }
        }
    }

    if args.no_details {
        finalize_progress_bar();
    }
    eprintln!("{}  {}/{} exact  RTFx={:.1}", label, exact, wavs.len(), total_audio_s / total_elapsed_s);

    Ok(())
}
