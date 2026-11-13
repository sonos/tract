use std::io::Read as _;
use std::io::Write as _;
use std::net::TcpListener;
use std::net::TcpStream;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::*;
use clap::Parser;
use nemotron_voicechat::FRAME_SECS;
use nemotron_voicechat::INPUT_RATE;
use nemotron_voicechat::Models;
use nemotron_voicechat::OUTPUT_RATE;
use nemotron_voicechat::Session;
use nemotron_voicechat::frame_stats;
use tungstenite::Message;
use tungstenite::WebSocket;

const PAGE: &str = include_str!("voicechat-page.html");

/// VoiceChat as a live voice service. Serves a web page that streams the browser's microphone
/// (16 kHz f32) over a websocket and plays the agent's voice (22.05 kHz) as it comes back, with
/// both transcripts. One conversation at a time, each starting fresh.
///
/// Text messages from the server: "\u{1}ready" once the session is set up, "\u{2}" + user
/// transcript, "\u{3}" + turn-taking rule fired, anything else is agent text.
#[derive(Parser, Debug)]
struct Args {
    /// Directory holding model/, as written by export.py.
    #[arg(long, default_value = "assets")]
    assets: String,
    #[arg(long, default_value = "0.0.0.0:8998")]
    listen: String,
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
    /// Record each conversation under this directory: transcript.txt (one line per 80 ms
    /// frame, written as it goes), in.wav and out.wav.
    #[arg(long)]
    record: Option<PathBuf>,
}

fn write_wav(path: &Path, rate: u32, samples: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    samples.iter().try_for_each(|s| writer.write_sample(*s))?;
    writer.finalize()?;
    Ok(())
}

struct Recording {
    dir: PathBuf,
    transcript: std::fs::File,
    input: Vec<f32>,
    output: Vec<f32>,
}

fn converse(
    models: &Models,
    args: &Args,
    ws: &mut WebSocket<TcpStream>,
    recording: &mut Option<Recording>,
) -> Result<()> {
    let start = Instant::now();
    let mut session = Session::new(models, args.seed, !args.no_turn_taking)?;
    eprintln!("session ready in {:.0} ms", start.elapsed().as_secs_f64() * 1000.0);
    ws.send(Message::Text("\u{1}ready".into()))?;
    let result = (|| -> Result<()> {
        loop {
            let pcm: Vec<f32> = match ws.read()? {
                Message::Binary(b) => {
                    b.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
                }
                Message::Close(_) => return Ok(()),
                _ => continue,
            };
            let first = session.frames().len();
            let (voice, frames) = session.push(&pcm)?;
            if !voice.is_empty() {
                ws.send(Message::Binary(
                    voice.iter().flat_map(|s| s.to_le_bytes()).collect::<Vec<u8>>().into(),
                ))?;
            }
            let agent: String = frames.iter().map(|f| models.piece(f.agent)).collect();
            let user: String = frames.iter().map(|f| f.user.as_str()).collect();
            if !agent.is_empty() {
                ws.send(Message::Text(agent.into()))?;
            }
            if !user.is_empty() {
                ws.send(Message::Text(format!("\u{2}{user}").into()))?;
            }
            for (i, frame) in frames.iter().enumerate() {
                let n = first + i;
                if let Some(rule) = frame.rule {
                    eprintln!("frame {n}: {rule}");
                    ws.send(Message::Text(
                        format!("\u{3}{:.2} s: {rule}", n as f64 * FRAME_SECS).into(),
                    ))?;
                }
                if let Some(r) = recording {
                    let mut line = format!(
                        "{n:5} {:7.2}s {} user={:<14} agent={:?}",
                        n as f64 * FRAME_SECS,
                        if frame.user_speech { '#' } else { '.' },
                        format!("{:?}", frame.user),
                        models.piece(frame.agent)
                    );
                    if let Some(said) = frame.model_said {
                        line += &format!(" (model said {:?})", models.piece(said));
                    }
                    if let Some(rule) = frame.rule {
                        line += &format!("  <<< {rule}");
                    }
                    writeln!(r.transcript, "{line}")?;
                }
            }
            if let Some(r) = recording {
                r.input.extend_from_slice(&pcm);
                r.output.extend_from_slice(&voice);
            }
        }
    })();
    eprintln!("session over: {}", frame_stats(session.frames()));
    result
}

fn serve(models: &Models, args: &Args, stream: TcpStream) -> Result<()> {
    let mut ws = tungstenite::accept(stream).map_err(|e| anyhow!("websocket handshake: {e}"))?;
    let mut recording = match &args.record {
        Some(root) => {
            let stamp =
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs();
            let dir = root.join(stamp.to_string());
            std::fs::create_dir_all(&dir)?;
            eprintln!("recording to {}", dir.display());
            let transcript = std::fs::File::create(dir.join("transcript.txt"))?;
            Some(Recording { dir, transcript, input: vec![], output: vec![] })
        }
        None => None,
    };
    let result = converse(models, args, &mut ws, &mut recording);
    if let Some(r) = recording {
        write_wav(&r.dir.join("in.wav"), INPUT_RATE, &r.input)?;
        write_wav(&r.dir.join("out.wav"), OUTPUT_RATE, &r.output)?;
    }
    result
}

fn main() -> Result<()> {
    let args = Args::parse();
    let runtime = tract::runtime_for_name(&args.runtime)?;
    let t0 = Instant::now();
    let models = Models::load(&format!("{}/model", args.assets), &runtime, args.q4)?;
    eprintln!("loaded in {:.1} s", t0.elapsed().as_secs_f64());
    let listener = TcpListener::bind(&args.listen)?;
    eprintln!("open http://{} in a browser", args.listen);
    for stream in listener.incoming() {
        let mut stream = stream?;
        let mut head = [0u8; 2048];
        let n = stream.peek(&mut head)?;
        if String::from_utf8_lossy(&head[..n]).to_ascii_lowercase().contains("upgrade: websocket") {
            if let Err(e) = serve(&models, &args, stream) {
                eprintln!("session ended: {e}");
            }
        } else {
            let _ = stream.read(&mut [0u8; 4096]);
            let reply = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/html; charset=utf-8\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{PAGE}",
                PAGE.len()
            );
            let _ = stream.write_all(reply.as_bytes());
        }
    }
    Ok(())
}
