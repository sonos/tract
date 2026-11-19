use std::io::Write;
use std::time::Duration;
use std::time::Instant;

use anyhow::*;
use clap::Parser;
use tract::prelude::*;

/// Streams a 16 kHz WAV through Voxtral Mini 4B Realtime one 80 ms chunk at a time: the
/// pulsed encoder turns each 1280-sample chunk into one audio embedding, the decoder adds it
/// to the embedding of the previous token and predicts the next one.
#[derive(Parser, Debug)]
struct Args {
    /// Directory holding model/ (and reference/ for --check), as written by export.py.
    #[arg(long, default_value = "assets")]
    assets: String,

    /// 16 kHz mono WAV to transcribe.
    #[arg(default_value = "assets/2086-149220-0033.wav")]
    input: String,

    /// Runtime to run both graphs on.
    #[arg(long, default_value = "default")]
    runtime: String,

    /// Run both graphs in f16 (the log-mel stays in f32).
    #[arg(long)]
    f16: bool,

    /// Quantize the decoder's weights to Q4_0 blocks.
    #[arg(long)]
    q4: bool,

    /// Compare the tokens with the reference export.py --check wrote for the sample.
    #[arg(long)]
    check: bool,
}

const CHUNK: usize = 1280;
/// The encoder reads its input this many samples early: see export.py.
const LOOK_AHEAD: usize = 40;
/// Silence after the audio, in tokens, past the delay: one, plus room for a last long word.
const TAIL_TOKENS: usize = 11;
const BOS: i64 = 1;
const STREAMING_PAD: i64 = 32;

#[derive(serde::Deserialize)]
struct Config {
    delay_tokens: usize,
    left_pad_tokens: usize,
    special_tokens: usize,
}

fn read_wav(path: &str) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    ensure!(spec.sample_rate == 16000 && spec.channels == 1, "{path}: expects 16 kHz mono");
    Ok(match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1u32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|s| s as f32 / scale))
                .collect::<Result<_, _>>()?
        }
    })
}

fn hex(s: &str) -> Result<Vec<u8>> {
    (0..s.len()).step_by(2).map(|i| Ok(u8::from_str_radix(&s[i..i + 2], 16)?)).collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    let dir = format!("{}/model", args.assets);
    let config: Config =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/config.json"))?)?;
    let vocab: Vec<String> =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/vocab.json"))?)?;
    let vocab = vocab.iter().map(|s| hex(s)).collect::<Result<Vec<_>>>()?;

    let t0 = Instant::now();
    let nnef = tract::nnef()?.with_tract_transformers()?;
    let runtime = tract::runtime_for_name(&args.runtime)?;
    let mut encoder = nnef.load(format!("{dir}/encoder.nnef.tgz"))?;
    encoder.transform(Pulse::new(CHUNK.to_string()).symbol("T"))?;
    ensure!(encoder.property("pulse.delay")?.as_slice::<i64>()?[0] == 0);
    if args.f16 {
        encoder.transform(r#"{"name":"f32_to_f16","keep_io":true,"exclude":["mel_"]}"#)?;
    }
    let mut encoder = runtime.prepare(encoder)?.spawn_state()?;
    let mut decoder = nnef.load(format!("{dir}/decoder.nnef.tgz"))?;
    decoder.transform("transformers_detect_all")?;
    if args.q4 {
        decoder.transform("block_quant")?;
    }
    if args.f16 {
        decoder.transform(r#"{"name":"f32_to_f16","keep_io":true}"#)?;
    }
    let mut decoder = runtime.prepare(decoder)?.spawn_state()?;
    eprintln!("loaded on {} in {:.1} s", runtime.name()?, t0.elapsed().as_secs_f64());

    // The stream opens on left_pad_tokens of silence, while the decoder reads its prompt:
    // BOS, then a streaming pad per token of that silence and of the delay.
    let audio = read_wav(&args.input)?;
    let mut stream = vec![0f32; config.left_pad_tokens * CHUNK - LOOK_AHEAD];
    stream.extend_from_slice(&audio);
    let steps = (stream.len() + LOOK_AHEAD).div_ceil(CHUNK) + config.delay_tokens + TAIL_TOKENS;
    stream.resize(steps * CHUNK, 0.0);
    let prompt = config.left_pad_tokens + config.delay_tokens;

    let mut tokens = vec![BOS];
    let mut timings: Vec<(Duration, Duration)> = vec![];
    for (t, chunk) in stream.chunks(CHUNK).take(steps - 1).enumerate() {
        let start = Instant::now();
        let embedding = encoder.run([Tensor::from_slice(&[1, 1, CHUNK], chunk)?])?.remove(0);
        let encoded = Instant::now();
        let token = Tensor::from_slice(&[1, 1], &[tokens[t]])?;
        let logits = decoder.run([token, embedding])?.remove(0);
        let logits = logits.as_slice::<f32>()?;
        let next = (0..logits.len()).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap();
        timings.push((encoded - start, encoded.elapsed()));
        let next = if t < prompt { STREAMING_PAD } else { next as i64 };
        tokens.push(next);
        if let Some(piece) = (next as usize).checked_sub(config.special_tokens) {
            std::io::stdout().write_all(&vocab[piece])?;
            std::io::stdout().flush()?;
        }
    }
    println!();

    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    let steady = &timings[1..];
    let mean = |pick: fn(&(Duration, Duration)) -> Duration| {
        ms(steady.iter().map(pick).sum::<Duration>()) / steady.len() as f64
    };
    let worst = steady.iter().map(|(e, d)| *e + *d).max().unwrap_or_default();
    eprintln!(
        "{} steps for {:.2} s of audio: first {:.1} ms, then encoder {:.2} + decoder {:.2} ms per 80 ms step, worst {:.2} ms",
        timings.len(),
        audio.len() as f64 / 16000.0,
        ms(timings[0].0 + timings[0].1),
        mean(|t| t.0),
        mean(|t| t.1),
        ms(worst)
    );

    if args.check {
        let path = format!("{}/reference/tokens.json", args.assets);
        let reference: Vec<i64> = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        let differ = (0..reference.len().max(tokens.len()))
            .filter(|&i| reference.get(i) != tokens.get(i))
            .count();
        eprintln!("{differ} of {} tokens differ from the reference", reference.len());
        ensure!(differ == 0, "the transcript differs from the reference");
    }
    Ok(())
}
