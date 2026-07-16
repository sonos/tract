//! Minimal generation client: send a tokenized prompt to a running Dis-tract
//! coordinator and print the generated token ids.
//!
//!   distract-gen `<comma,separated,prompt,ids> [max_tokens]`

use anyhow::{Result, anyhow};
use std::time::Duration;
use tract_distributed::protocol::{GenerateReply, GenerateRequest};
use tract_distributed::znet;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let prompt: Vec<i64> = std::env::args()
        .nth(1)
        .expect("prompt ids")
        .split(',')
        .filter_map(|x| x.trim().parse().ok())
        .collect();
    let max_tokens: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(16);

    let session = zenoh::open(znet::worker_config()?).await.map_err(znet::zerr)?;
    let stop: Vec<i64> = std::env::args()
        .nth(3)
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_default();
    let req = GenerateRequest { prompt, max_tokens, stream_id: 0, stop };
    let replies = session
        .get(znet::GENERATE_KEY)
        .payload(serde_json::to_vec(&req)?)
        .timeout(Duration::from_secs(600))
        .await
        .map_err(znet::zerr)?;
    let reply = replies.recv_async().await.map_err(|e| anyhow!("{e:?}"))?;
    let sample = reply.result().map_err(|e| anyhow!("{e:?}"))?;
    let r: GenerateReply = serde_json::from_slice(&sample.payload().to_bytes())?;
    println!("TOKENS={}", r.tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","));
    eprintln!("ttft={:.0}ms decode={:.1}tok/s", r.ttft_ms, r.decode_tok_s);
    Ok(())
}
