use std::time::Instant;

use anyhow::*;
use clap::Parser;
use kyutai_moshi::{FRAME, Models, SAMPLE_RATE};
use tract::prelude::*;

/// Talks to Moshi through WAV files: the input is the user's side of the conversation,
/// Moshi's voice goes to the output, its words to the terminal. With --check, replays the
/// user codes of Kyutai's greedy reference (export.py --check) and fails on any token that
/// differs.
#[derive(Parser, Debug)]
struct Args {
    /// Directory holding model/ (and reference/ for --check), as written by export.py.
    #[arg(long, default_value = "assets")]
    assets: String,
    /// 24 kHz mono WAV of the user's side.
    #[arg(long)]
    input: Option<String>,
    /// Where to write Moshi's voice.
    #[arg(long, default_value = "moshi.wav")]
    output: String,
    /// Compare greedy generation with Kyutai's reference instead of sampling.
    #[arg(long)]
    check: bool,
    #[arg(long, default_value = "cuda")]
    runtime: String,
    #[arg(long, default_value_t = 1)]
    seed: u64,
}

fn read_wav(path: &str) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    ensure!(reader.spec().sample_rate == SAMPLE_RATE as u32 && reader.spec().channels == 1);
    Ok(match reader.spec().sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<Result<_, _>>()?,
    })
}

fn tokens(value: &serde_json::Value) -> Option<Vec<i64>> {
    value.as_array().map(|a| a.iter().map(|v| v.as_i64().unwrap()).collect())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let t0 = Instant::now();
    let runtime = tract::runtime_for_name(&args.runtime)?;
    let models = Models::load(&format!("{}/model", args.assets), &runtime)?;
    eprintln!("loaded on {} in {:.1}s", runtime.name()?, t0.elapsed().as_secs_f64());

    let reference: Option<serde_json::Value> = if args.check {
        let path = format!("{}/reference/moshi_greedy.json", args.assets);
        Some(serde_json::from_str(&std::fs::read_to_string(&path).context(path)?)?)
    } else {
        None
    };
    let input = args.input.as_deref().map(read_wav).transpose()?;
    let frames = match (&reference, &input) {
        (Some(r), _) => r["outputs"].as_array().unwrap().len(),
        (None, Some(wav)) => wav.len() / FRAME,
        (None, None) => bail!("pass --input, or --check"),
    };

    let mut session = models.session(args.seed, args.check)?;
    let mut voice: Vec<f32> = vec![];
    let mut text = String::new();
    let mut mismatches = 0;
    let mut times = vec![];
    for t in 0..frames {
        let start = Instant::now();
        let frame = if let Some(r) = &reference {
            let user: [i64; 8] = std::array::from_fn(|c| r["user_codes"][c][t].as_i64().unwrap());
            session.step_codes(&user)?
        } else {
            session.step_audio(&input.as_ref().unwrap()[t * FRAME..][..FRAME])?
        };
        times.push(start.elapsed().as_secs_f64() * 1000.0);
        if let Some(piece) = &frame.text {
            text.push_str(piece);
        }
        voice.extend_from_slice(&frame.audio);
        if let Some(r) = &reference {
            let expected = tokens(&r["outputs"][t]);
            if expected != frame.tokens {
                mismatches += 1;
                println!("frame {t}: tract {:?}, kyutai {expected:?}", frame.tokens);
            }
        }
    }

    let mut steady: Vec<f64> = times.iter().skip(2).cloned().collect();
    steady.sort_by(f64::total_cmp);
    if let (Some(median), Some(worst)) = (steady.get(steady.len() / 2), steady.last()) {
        println!(
            "{frames} frames: first {:.0} ms, then median {median:.1} ms, worst {worst:.1} ms (80 ms per frame for real time)",
            times[0]
        );
    }
    println!("moshi: {}", text.trim());
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE as u32,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(&args.output, spec)?;
    voice.iter().try_for_each(|s| writer.write_sample(*s))?;
    writer.finalize()?;
    if args.check {
        println!("{mismatches} of {frames} frames differ from Kyutai's greedy LMGen");
        ensure!(mismatches == 0, "tract and Kyutai's reference disagree");
    }
    Ok(())
}
