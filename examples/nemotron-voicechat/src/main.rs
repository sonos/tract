use std::time::Instant;

use anyhow::*;
use clap::Parser;
use nemotron_voicechat::BOS;
use nemotron_voicechat::EOS;
use nemotron_voicechat::FRAME_SECS;
use nemotron_voicechat::Frame;
use nemotron_voicechat::INPUT_RATE;
use nemotron_voicechat::Models;
use nemotron_voicechat::OUTPUT_RATE;
use nemotron_voicechat::Session;
use nemotron_voicechat::frame_stats;
use nemotron_voicechat::resample;

/// Talks to VoiceChat through WAV files: the input is the user's side of the conversation, fed
/// 80 ms at a time as a microphone would. Writes a stereo WAV with the user on the left and the
/// agent on the right, and prints both transcripts and the turns.
#[derive(Parser, Debug)]
struct Args {
    /// Directory holding model/, as written by export.py.
    #[arg(long, default_value = "assets")]
    assets: String,
    /// The user's side; the first channel is used, at any sample rate.
    #[arg(long)]
    input: String,
    #[arg(long, default_value = "voicechat.wav")]
    output: String,
    #[arg(long, default_value = "cuda")]
    runtime: String,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Quantize the language model to Q4_0 at load time.
    #[arg(long)]
    q4: bool,
    /// Run the model alone, without NVIDIA's turn-taking rules.
    #[arg(long)]
    no_turn_taking: bool,
}

fn read_first_channel(path: &str) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|s| s as f32 / scale))
                .collect::<Result<_, _>>()?
        }
    };
    Ok((samples.into_iter().step_by(spec.channels as usize).collect(), spec.sample_rate))
}

/// User speech spans (RNNT speech frames, gaps under 0.4 s merged) and agent turns, with the
/// delay from the end of the user's speech to each agent turn.
fn timeline(frames: &[Frame]) -> Vec<String> {
    let secs = |f: usize| f as f64 * FRAME_SECS;
    let mut spans: Vec<(usize, usize)> = vec![];
    for (f, _) in frames.iter().enumerate().filter(|(_, f)| f.user_speech) {
        match spans.last_mut() {
            Some(last) if f - last.1 <= 5 => last.1 = f,
            _ => spans.push((f, f)),
        }
    }
    let mut lines: Vec<(usize, String)> = spans
        .iter()
        .map(|&(a, b)| (a, format!("user  speaks {:6.2} - {:6.2} s", secs(a), secs(b + 1))))
        .collect();
    for (f, frame) in frames.iter().enumerate() {
        let rule = frame.rule.map_or(String::new(), |r| format!("  [{r}]"));
        if frame.agent == BOS {
            let delay = spans.iter().rev().find(|s| s.1 < f).map_or(String::new(), |s| {
                format!(", {:.0} ms after the user", secs(f - s.1 - 1) * 1000.0)
            });
            lines.push((f, format!("agent starts {:6.2} s{delay}{rule}", secs(f))));
        } else if frame.agent == EOS {
            lines.push((f, format!("agent stops  {:6.2} s{rule}", secs(f))));
        }
    }
    lines.sort_by_key(|l| l.0);
    lines.into_iter().map(|l| l.1).collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    let (pcm, rate) = read_first_channel(&args.input)?;
    let pcm = resample(&pcm, rate, INPUT_RATE);
    let runtime = tract::runtime_for_name(&args.runtime)?;
    let t0 = Instant::now();
    let models = Models::load(&format!("{}/model", args.assets), &runtime, args.q4)?;
    eprintln!("loaded in {:.1} s", t0.elapsed().as_secs_f64());
    let mut session = Session::new(&models, args.seed, !args.no_turn_taking)?;
    let mut voice = vec![];
    let mut agent = String::new();
    // 320 ms of silence flushes the frames still in the mel and encoder pipelines.
    let flushed =
        pcm.iter().copied().chain(std::iter::repeat_n(0.0, 4 * 1280)).collect::<Vec<f32>>();
    for chunk in flushed.chunks(1280) {
        let (wav, frames) = session.push(chunk)?;
        voice.extend(wav);
        for frame in frames {
            agent += models.piece(frame.agent);
        }
    }
    let frames = session.frames();
    let user: String = frames.iter().map(|f| f.user.as_str()).collect();
    println!("{}", frame_stats(frames));
    println!("user (RNNT): {}", user.trim());
    println!("agent: {}", agent.trim());
    for line in timeline(frames) {
        println!("{line}");
    }
    let user_out = resample(&pcm, INPUT_RATE, OUTPUT_RATE);
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: OUTPUT_RATE,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(&args.output, spec)?;
    for i in 0..user_out.len().max(voice.len()) {
        writer.write_sample(user_out.get(i).copied().unwrap_or(0.0))?;
        writer.write_sample(voice.get(i).copied().unwrap_or(0.0))?;
    }
    writer.finalize()?;
    Ok(())
}
