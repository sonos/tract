use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::*;
use clap::Parser;
use float_ord::FloatOrd;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use itertools::Itertools;
use tract_rs::prelude::*;
use tract_rs::Nnef;

mod greedy;
mod beam;
mod fbsd;

#[derive(clap::ValueEnum, Clone)]
enum Decoder { Greedy, Beam, Fbsd }

#[derive(Parser)]
#[command(about = "NeMo Parakeet ASR inference")]
struct Args {
    #[command(flatten)]
    beam: beam::BeamConfig,
    #[command(flatten)]
    fbsd: fbsd::FbsdConfig,
    #[arg(long, value_enum, default_value_t = Decoder::Greedy)]
    decoder: Decoder,
    #[arg(long)]
    stats: bool,
    #[arg(long)]
    no_details: bool,
    /// Run ground-truth decoder and write transcript to a .txt file beside each wav
    #[arg(long)]
    write_gt: bool,
    /// Sweep hardcoded decoder configs and print TSV results to stdout
    #[arg(long)]
    param_search: bool,
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
}

#[derive(Default)]
pub(crate) struct CallStats {
    calls: u32,
    total_batch: u64,
    pub(crate) total_us: u64,
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
        Decoder::Fbsd => format!("fbsd  beam_size={}  beam_dur_k={}  max_symbols={}", args.fbsd.fbsd_beam_size, args.fbsd.fbsd_beam_dur_k, args.fbsd.fbsd_max_symbols_per_frame),
    }
}

fn run_decoder(model: &TdtModel, wav: &[f32], args: &Args) -> Result<(String, DecodingStats)> {
    match args.decoder {
        Decoder::Greedy => greedy::transcribe_greedy(model, wav),
        Decoder::Beam   => beam::transcribe_beam(model, wav, &args.beam),
        Decoder::Fbsd   => fbsd::transcribe_fbsd(model, wav, &args.fbsd),
    }
}

// ─── SearchConfig ────────────────────────────────────────────────────────────

enum SearchConfig {
    Greedy,
    Beam(beam::BeamConfig),
    Fbsd(fbsd::FbsdConfig),
}

impl SearchConfig {
    fn label(&self) -> String {
        match self {
            SearchConfig::Greedy => "greedy".to_owned(),
            SearchConfig::Beam(c) => format!("beam_{}_{}", c.beam_size, c.beam_dur_k),
            SearchConfig::Fbsd(c) => format!("fbsd_{}_{}_{}", c.fbsd_beam_size, c.fbsd_beam_dur_k, c.fbsd_max_symbols_per_frame),
        }
    }

    fn run(&self, model: &TdtModel, wav: &[f32]) -> Result<(String, DecodingStats)> {
        match self {
            SearchConfig::Greedy    => greedy::transcribe_greedy(model, wav),
            SearchConfig::Beam(c)   => beam::transcribe_beam(model, wav, c),
            SearchConfig::Fbsd(c)   => fbsd::transcribe_fbsd(model, wav, c),
        }
    }
}

fn search_configs() -> Vec<SearchConfig> {
    fn b(beam_size: usize, beam_dur_k: usize) -> SearchConfig {
        SearchConfig::Beam(beam::BeamConfig { beam_size, beam_dur_k })
    }
    fn a(beam_size: usize, beam_dur_k: usize, max_symbols: usize) -> SearchConfig {
        SearchConfig::Fbsd(fbsd::FbsdConfig { fbsd_beam_size: beam_size, fbsd_beam_dur_k: beam_dur_k, fbsd_max_symbols_per_frame: max_symbols })
    }
    vec![
        SearchConfig::Greedy,
        b(1, 1),
        b(2, 1),
        b(2, 2),
        b(4, 1),
        b(4, 2),
        b(4, 4),
        b(8, 2),
        b(8, 4),
        a(1, 1, 10),
        a(2, 1, 10),
        a(2, 2, 10),
        a(4, 1, 10),
        a(4, 2, 10),
        a(4, 4, 10),
        a(8, 2, 10),
        a(8, 4, 10),
        a(4, 2, 3),
        a(4, 2, 30),
    ]
}

fn param_search(model: &TdtModel, wavs: &[PathBuf]) -> Result<()> {
    // Warmup
    if let Some(first) = wavs.first() {
        let (wav, _) = load_wav(first)?;
        greedy::transcribe_greedy(model, &wav)?;
    }

    let configs = search_configs();

    let mp = MultiProgress::new();
    let cfg_style = ProgressStyle::with_template(
        "Configs  {bar:40} {pos:>3}/{len} {msg}"
    ).unwrap();
    let file_style = ProgressStyle::with_template(
        "Files    {bar:40} {pos:>3}/{len}"
    ).unwrap();

    let cfg_bar = mp.add(ProgressBar::new(configs.len() as u64));
    cfg_bar.set_style(cfg_style);
    let file_bar = mp.add(ProgressBar::new(wavs.len() as u64));
    file_bar.set_style(file_style);

    mp.suspend(|| println!("label\tEPR\tRTFx\tpre%\tenc%\tdec%\tjoint%\thost%"));

    for cfg in &configs {
        let label = cfg.label();
        cfg_bar.set_message(label.clone());

        file_bar.reset();
        file_bar.set_length(wavs.len() as u64);

        let mut total_audio_s = 0.0f64;
        let mut total_elapsed_s = 0.0f64;
        let mut exact = 0usize;
        let mut total = 0usize;
        let mut pre_us = 0u64;
        let mut enc_us = 0u64;
        let mut dec_us = 0u64;
        let mut joint_us = 0u64;

        for wav_path in wavs {
            let (wav, audio_s) = load_wav(wav_path)?;
            let reference = std::fs::read_to_string(wav_path.with_extension("txt"))
                .with_context(|| format!("no ground-truth transcript for {} (run with --write-gt first)", wav_path.display()))?;
            let reference = reference.trim_end_matches('\n').to_owned();

            let t = Instant::now();
            let (transcript, stats) = cfg.run(model, &wav)?;
            let elapsed = t.elapsed().as_secs_f64();

            total_audio_s += audio_s;
            total_elapsed_s += elapsed;
            total += 1;
            if clean(&transcript) == reference { exact += 1; }
            pre_us   += stats.preprocessor.total_us;
            enc_us   += stats.encoder.total_us;
            dec_us   += stats.decoder.total_us;
            joint_us += stats.joint.total_us;

            file_bar.inc(1);
        }

        let epr = if total > 0 { exact as f64 / total as f64 } else { 0.0 };
        let rtfx = if total_elapsed_s > 0.0 { total_audio_s / total_elapsed_s } else { 0.0 };
        let total_us = (total_elapsed_s * 1_000_000.0) as u64;
        let pct = |us: u64| if total_us > 0 { us as f64 / total_us as f64 * 100.0 } else { 0.0 };
        let nn_us = pre_us + enc_us + dec_us + joint_us;
        let host_us = total_us.saturating_sub(nn_us);
        mp.suspend(|| println!("{}\t{:.4}\t{:.1}\t{:.1}\t{:.1}\t{:.1}\t{:.1}\t{:.1}",
            label, epr, rtfx,
            pct(pre_us), pct(enc_us), pct(dec_us), pct(joint_us), pct(host_us)));

        cfg_bar.inc(1);
    }

    cfg_bar.finish();
    file_bar.finish();

    Ok(())
}

// ─── write_gt ────────────────────────────────────────────────────────────────

fn write_gt(model: &TdtModel, wavs: &[PathBuf], no_details: bool) -> anyhow::Result<()> {
    let gt_cfg = beam::BeamConfig { beam_size: 10, beam_dur_k: 5 };
    // Warmup
    if let Some(first) = wavs.first() {
        let (wav, _) = load_wav(first)?;
        beam::transcribe_beam(model, &wav, &gt_cfg)?;
    }
    let pb = if no_details {
        let pb = ProgressBar::new(wavs.len() as u64);
        pb.set_style(
            ProgressStyle::with_template("Writing GT  {bar:40} {pos:>3}/{len}").unwrap()
        );
        Some(pb)
    } else {
        None
    };
    for wav_path in wavs {
        let (wav, _) = load_wav(wav_path)?;
        let (transcript, _) = beam::transcribe_beam(model, &wav, &gt_cfg)?;
        let txt_path = wav_path.with_extension("txt");
        std::fs::write(&txt_path, clean(&transcript))?;
        if let Some(ref pb) = pb {
            pb.inc(1);
        } else {
            eprintln!("{} -> {}", wav_path.display(), txt_path.display());
        }
    }
    if let Some(pb) = pb { pb.finish(); }
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

    if args.param_search {
        return param_search(&model, &wavs);
    }

    let label = decoder_label(&args);
    if !args.no_details {
        eprintln!("{}  files={}", label, wavs.len());
    }

    // Warmup on first file
    if let Some(first) = wavs.first() {
        let (wav, _) = load_wav(first)?;
        run_decoder(&model, &wav, &args)?;
    }

    let pb = if args.no_details {
        let pb = ProgressBar::new(wavs.len() as u64);
        pb.set_style(
            ProgressStyle::with_template("Decoding  {bar:40} {pos:>3}/{len}").unwrap()
        );
        Some(pb)
    } else {
        None
    };

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

        let ok = clean(&transcript) == reference.trim_end_matches('\n');
        if ok { exact += 1; }

        if let Some(ref pb) = pb {
            pb.inc(1);
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

    if let Some(pb) = pb { pb.finish(); }
    eprintln!("{}  {}/{} exact  RTFx={:.1}", label, exact, wavs.len(), total_audio_s / total_elapsed_s);

    Ok(())
}
