use std::time::Duration;
use std::time::Instant;

use anyhow::*;
use clap::Parser;
use tract::prelude::*;

/// Streams a 24 kHz WAV through kyutai/mimi one 80 ms frame at a time: the
/// pulsed encoder turns each 1920-sample frame into 8 codes, the pulsed
/// decoder turns them back into 1920 samples. Checks codes and audio against
/// the PyTorch reference written by export.py.
#[derive(Parser, Debug)]
struct Args {
    /// Directory holding model/ and reference/, as written by export.py.
    #[arg(long, default_value = "assets")]
    assets: String,

    /// Runtime to run both graphs on.
    #[arg(long, default_value = "default")]
    runtime: String,

    /// Where to write the streamed reconstruction.
    #[arg(long, default_value = "assets/recon-tract.wav")]
    output: String,
}

const FRAME: usize = 1920;
const CODEBOOKS: usize = 8;

struct Streamer {
    state: State,
    delay: usize,
}

fn load(
    nnef: &Nnef,
    runtime: &Runtime,
    path: &str,
    symbol: &str,
    pulse: usize,
) -> Result<Streamer> {
    let mut model = nnef.load(path)?;
    model.transform(Pulse::new(pulse.to_string()).symbol(symbol))?;
    let delay = model.property("pulse.delay")?.as_slice::<i64>()?[0] as usize;
    let state = runtime.prepare(model)?.spawn_state()?;
    Ok(Streamer { state, delay })
}

fn read_wav(path: &str) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    ensure!(reader.spec().sample_rate == 24000 && reader.spec().channels == 1);
    Ok(reader.samples::<f32>().collect::<Result<_, _>>()?)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let nnef = tract::nnef()?.with_tract_transformers()?;
    let runtime = tract::runtime_for_name(&args.runtime)?;
    let assets = &args.assets;
    let mut encoder =
        load(&nnef, &runtime, &format!("{assets}/model/mimi-encoder.nnef.tgz"), "T", FRAME)?;
    let mut decoder =
        load(&nnef, &runtime, &format!("{assets}/model/mimi-decoder.nnef.tgz"), "F", 1)?;
    eprintln!(
        "runtime {}, encoder delay {} frames, decoder delay {} samples",
        runtime.name()?,
        encoder.delay,
        decoder.delay
    );

    let input = read_wav(&format!("{assets}/reference/input.wav"))?;
    let reference: Vec<Vec<i64>> =
        serde_json::from_str(&std::fs::read_to_string(format!("{assets}/reference/codes.json"))?)?;
    let frames = input.len() / FRAME;

    let mut codes: Vec<[i64; CODEBOOKS]> = vec![];
    let mut audio: Vec<f32> = vec![];
    let mut timings: Vec<(Duration, Duration)> = vec![];
    let silence = vec![0f32; FRAME];
    for f in 0..frames + encoder.delay {
        let chunk = input.get(f * FRAME..(f + 1) * FRAME).unwrap_or(&silence);
        let start = Instant::now();
        let out = encoder.state.run([Tensor::from_slice(&[1, 1, FRAME], chunk)?])?;
        let enc_done = Instant::now();
        if f < encoder.delay {
            continue;
        }
        let frame_codes: [i64; CODEBOOKS] = out[0].as_slice::<i64>()?.try_into()?;
        codes.push(frame_codes);
        let out = decoder.state.run([Tensor::from_slice(&[1, CODEBOOKS, 1], &frame_codes)?])?;
        timings.push((enc_done - start, enc_done.elapsed()));
        audio.extend_from_slice(out[0].as_slice::<f32>()?);
    }
    let audio = &audio[decoder.delay.min(audio.len())..];

    let mismatches: usize = (0..CODEBOOKS)
        .map(|k| (0..frames).filter(|&f| reference[k][f] != codes[f][k]).count())
        .sum();
    let torch = read_wav(&format!("{assets}/reference/recon.wav"))?;
    let compared = torch.len().min(audio.len());
    let max_diff = (0..compared).map(|i| (torch[i] - audio[i]).abs()).fold(0f32, f32::max);
    let rms = (torch.iter().map(|x| x * x).sum::<f32>() / torch.len() as f32).sqrt();

    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    let steady = &timings[1..];
    let mean = |pick: fn(&(Duration, Duration)) -> Duration| {
        ms(steady.iter().map(pick).sum::<Duration>()) / steady.len() as f64
    };
    let worst = steady.iter().map(|(e, d)| *e + *d).max().unwrap_or_default();
    println!(
        "{frames} frames ({:.2} s): first frame {:.1} ms, then encode {:.2} + decode {:.2} ms/frame, worst {:.2} ms (budget 80 ms)",
        frames as f64 * 0.08,
        ms(timings[0].0 + timings[0].1),
        mean(|t| t.0),
        mean(|t| t.1),
        ms(worst)
    );
    println!(
        "codes: {mismatches} / {} differ from PyTorch; audio: max abs diff {max_diff:.2e} (rms {rms:.2e})",
        frames * CODEBOOKS
    );

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(&args.output, spec)?;
    audio.iter().try_for_each(|s| writer.write_sample(*s))?;
    writer.finalize()?;

    ensure!(mismatches == 0, "streamed codes differ from the PyTorch reference");
    ensure!(compared == frames * FRAME && max_diff < 1e-3, "streamed audio drifts from PyTorch");
    Ok(())
}
