use std::io::BufReader;
use std::io::BufWriter;
use std::io::Write;
use std::net::TcpListener;
use std::time::Instant;

use anyhow::*;
use clap::Parser;
use kyutai_moshi::{FRAME, Models, wire};

/// Serves Moshi over TCP, one conversation at a time: the client streams 80 ms frames of
/// 24 kHz audio and gets Moshi's text and voice back for each (see `moshi::wire`).
#[derive(Parser, Debug)]
struct Args {
    /// Directory holding model/, as written by export.py.
    #[arg(long, default_value = "assets")]
    assets: String,
    #[arg(long, default_value = "cuda")]
    runtime: String,
    #[arg(long, default_value = "0.0.0.0:8998")]
    listen: String,
    /// Quantize Helium and the depformer to Q4_0 at load time.
    #[arg(long)]
    q4: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let t0 = Instant::now();
    let runtime = tract::runtime_for_name(&args.runtime)?;
    let models = Models::load(&format!("{}/model", args.assets), &runtime, args.q4)?;
    let mut warmup = models.session(1, false)?;
    for _ in 0..4 {
        warmup.step_audio(&[0f32; FRAME])?;
    }
    drop(warmup);
    eprintln!("ready in {:.1}s, listening on {}", t0.elapsed().as_secs_f64(), args.listen);

    let listener = TcpListener::bind(&args.listen)?;
    for (n, stream) in listener.incoming().enumerate() {
        let stream = stream?;
        stream.set_nodelay(true)?;
        eprintln!("\nconversation {n} with {}", stream.peer_addr()?);
        let mut reader = BufReader::new(stream.try_clone()?);
        let mut writer = BufWriter::new(stream);
        let mut session = models.session(n as u64 + 1, false)?;
        let mut worst = 0f64;
        let mut frames = 0;
        let result: Result<()> = (|| loop {
            let pcm = match wire::read_frame(&mut reader) {
                Result::Ok(pcm) => pcm,
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
                Err(e) => return Err(e.into()),
            };
            let t = Instant::now();
            let frame = session.step_audio(&pcm)?;
            worst = worst.max(t.elapsed().as_secs_f64() * 1000.0);
            frames += 1;
            let text = frame.text.unwrap_or_default();
            if !text.is_empty() {
                eprint!("{text}");
            }
            wire::write_reply(&mut writer, &text, &frame.audio)?;
            writer.flush()?;
        })();
        eprintln!(
            "\nconversation {n} ended after {frames} frames, worst frame {worst:.1} ms: {result:?}"
        );
    }
    Ok(())
}
