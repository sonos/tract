use std::fs::File;
use std::path::Path;

use anyhow::*;
use clap::Parser;
use float_ord::FloatOrd;
use itertools::Itertools;
use tract_rs::prelude::*;
use tract_rs::Nnef;

mod greedy;
mod beam;

#[derive(Parser)]
#[command(about = "NeMo Parakeet ASR inference")]
struct Args {
    #[command(flatten)]
    beam: beam::BeamConfig,
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

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let nnef = tract_rs::nnef()?.with_tract_core()?.with_tract_transformers()?;
    let gpu = ["cuda", "metal", "default"]
        .iter()
        .find_map(|rt| tract_rs::runtime_for_name(rt).ok())
        .unwrap();

    let model = TdtModel::load(Path::new("assets/model"), &nnef, &gpu)?;

    let wav: Vec<f32> = hound::WavReader::open("assets/2086-149220-0033.wav")?
        .samples::<i16>()
        .map(|x| x.unwrap() as f32)
        .collect();

    greedy::transcribe_greedy(&model, &wav)?;
    let (transcript_g, dec, joint) = greedy::transcribe_greedy(&model, &wav)?;
    eprintln!("[greedy][decoder] {dec:?}");
    eprintln!("[greedy][joint]   {joint:?}");

    beam::transcribe_beam(&model, &wav, &args.beam)?;
    let (transcript_b, dec, joint) = beam::transcribe_beam(&model, &wav, &args.beam)?;
    eprintln!("[beam][decoder]   {dec:?}");
    eprintln!("[beam][joint]     {joint:?}");

    println!("Greedy: {transcript_g}");
    println!("Beam:   {transcript_b}");
    assert_eq!(
        transcript_g,
        "▁Well,▁I▁don't▁wish▁to▁see▁it▁any▁more,▁observed▁Phoebe,▁turning▁away▁her▁eyes."
    );
    assert_eq!(
        transcript_b,
        "▁Well,▁I▁don't▁wish▁to▁see▁it▁any▁more,▁observed▁Phoebe,▁turning▁away▁her▁eyes."
    );
    Ok(())
}
