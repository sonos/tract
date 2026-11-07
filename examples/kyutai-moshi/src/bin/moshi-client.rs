use std::collections::VecDeque;
use std::io::BufReader;
use std::io::BufWriter;
use std::io::Write;
use std::net::TcpStream;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::mpsc;
use std::time::Duration;
use std::time::Instant;

use anyhow::*;
use clap::Parser;
use kyutai_moshi::{FRAME, SAMPLE_RATE, wire};

/// Talks to a moshi-server: streams the microphone (or a WAV, paced in real time) as 80 ms
/// frames and plays Moshi's voice (or writes it to a WAV), printing its words as they come.
#[derive(Parser, Debug)]
struct Args {
    /// host:port of the moshi-server.
    #[arg(long)]
    server: String,
    /// Stream this 24 kHz mono WAV instead of the microphone.
    #[arg(long)]
    input: Option<String>,
    /// Write Moshi's voice to this WAV instead of the speakers.
    #[arg(long)]
    output: Option<String>,
}

type Queue = Arc<Mutex<VecDeque<f32>>>;

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

#[cfg(feature = "live")]
mod live {
    use super::*;
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use kyutai_moshi::Resampler;

    /// Microphone frames at 24 kHz, downmixed to mono.
    pub fn microphone(frames: mpsc::SyncSender<Vec<f32>>) -> Result<cpal::Stream> {
        let device = cpal::default_host().default_input_device().context("no microphone")?;
        let config = device.default_input_config()?;
        eprintln!("microphone: {} at {} Hz", device.description()?.name(), config.sample_rate());
        ensure!(config.sample_format() == cpal::SampleFormat::F32, "microphone is not f32");
        let channels = config.channels() as usize;
        let mut resampler = Resampler::new(config.sample_rate() as f64, SAMPLE_RATE as f64);
        let mut pending: Vec<f32> = vec![];
        let stream = device.build_input_stream(
            config.into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mono: Vec<f32> = data
                    .chunks(channels)
                    .map(|c| c.iter().sum::<f32>() / channels as f32)
                    .collect();
                resampler.push(&mono, &mut pending);
                while pending.len() >= FRAME {
                    let _ = frames.try_send(pending.drain(..FRAME).collect());
                }
            },
            |err| eprintln!("microphone error: {err}"),
            None,
        )?;
        stream.play()?;
        Ok(stream)
    }

    /// Plays the queue (24 kHz mono) on the default output, silence when it runs dry.
    pub fn speakers(queue: Queue) -> Result<cpal::Stream> {
        let device = cpal::default_host().default_output_device().context("no speakers")?;
        let config = device.default_output_config()?;
        eprintln!("speakers: {} at {} Hz", device.description()?.name(), config.sample_rate());
        ensure!(config.sample_format() == cpal::SampleFormat::F32, "speakers are not f32");
        let channels = config.channels() as usize;
        let mut resampler = Resampler::new(SAMPLE_RATE as f64, config.sample_rate() as f64);
        let mut ready: VecDeque<f32> = VecDeque::new();
        let stream = device.build_output_stream(
            config.into(),
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let wanted = data.len() / channels;
                if ready.len() < wanted {
                    let mut q = queue.lock().unwrap();
                    let take: Vec<f32> = q.drain(..).collect();
                    drop(q);
                    let mut out = vec![];
                    resampler.push(&take, &mut out);
                    ready.extend(out);
                }
                for frame in data.chunks_mut(channels) {
                    let s = ready.pop_front().unwrap_or(0.0);
                    frame.iter_mut().for_each(|x| *x = s);
                }
            },
            |err| eprintln!("speaker error: {err}"),
            None,
        )?;
        stream.play()?;
        Ok(stream)
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let stream = TcpStream::connect(&args.server)
        .with_context(|| format!("connecting to {}", args.server))?;
    stream.set_nodelay(true)?;
    eprintln!("connected to {}", args.server);
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = BufWriter::new(stream);

    let (tx, rx) = mpsc::sync_channel::<Vec<f32>>(16);
    let _mic: Option<Box<dyn std::any::Any>> = match &args.input {
        Some(path) => {
            let wav = read_wav(path)?;
            std::thread::spawn(move || {
                let start = Instant::now();
                for (i, frame) in wav.as_chunks::<FRAME>().0.iter().enumerate() {
                    let due = start + Duration::from_millis(80 * i as u64);
                    std::thread::sleep(due.saturating_duration_since(Instant::now()));
                    if tx.send(frame.to_vec()).is_err() {
                        break;
                    }
                }
            });
            None
        }
        None => {
            #[cfg(feature = "live")]
            {
                Some(Box::new(live::microphone(tx)?))
            }
            #[cfg(not(feature = "live"))]
            bail!("the microphone needs the `live` feature, or pass --input")
        }
    };

    let queue: Queue = Arc::new(Mutex::new(VecDeque::new()));
    let recorded: Queue = Arc::new(Mutex::new(VecDeque::new()));
    #[cfg(feature = "live")]
    let _speakers = if args.output.is_none() { Some(live::speakers(queue.clone())?) } else { None };
    #[cfg(not(feature = "live"))]
    ensure!(args.output.is_some(), "the speakers need the `live` feature, or pass --output");

    let sink = if args.output.is_some() { recorded.clone() } else { queue.clone() };
    let replies = std::thread::spawn(move || -> Result<usize> {
        let mut frames = 0;
        loop {
            let (text, audio) = match wire::read_reply(&mut reader) {
                Result::Ok(r) => r,
                Err(_) => return Ok(frames),
            };
            frames += 1;
            if !text.is_empty() {
                print!("{text}");
                std::io::stdout().flush()?;
            }
            sink.lock().unwrap().extend(audio);
        }
    });

    for frame in rx {
        wire::write_frame(&mut writer, &frame)?;
        writer.flush()?;
    }
    writer.get_ref().shutdown(std::net::Shutdown::Write)?;
    let frames = replies.join().unwrap()?;
    println!();
    eprintln!("{frames} frames answered");
    if let Some(path) = &args.output {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE as u32,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(path, spec)?;
        recorded.lock().unwrap().iter().try_for_each(|s| writer.write_sample(*s))?;
        writer.finalize()?;
    }
    Ok(())
}
