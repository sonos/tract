//! Measure what zenoh does with shard-sized payloads, to decide whether the shard
//! cache can serve one tensor per key.
//!
//! Serves `size_mb` of real bytes from a queryable and pulls them back through a
//! router on its own port, reporting throughput and whether they survived intact.
//!
//! `distract-wiretest [size_mb]`

use std::time::Instant;

use anyhow::{Result, anyhow};
use zenoh::Wait;

const ENDPOINT: &str = r#"["tcp/127.0.0.1:7459"]"#;
const KEY: &str = "distract/wiretest";

fn config(mode: &str, key: &str) -> Result<zenoh::Config> {
    let mut c = zenoh::Config::default();
    c.insert_json5("mode", &format!("\"{mode}\"")).map_err(|e| anyhow!("{e}"))?;
    c.insert_json5(key, ENDPOINT).map_err(|e| anyhow!("{e}"))?;
    Ok(c)
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let mb: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(333);
    let len = mb * 1024 * 1024;

    // Non-uniform bytes, so a truncated or zero-filled reply cannot pass as intact.
    let payload: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
    let expect_sum: u64 = payload.iter().map(|&b| b as u64).sum();

    let router =
        zenoh::open(config("router", "listen/endpoints")?).await.map_err(|e| anyhow!("{e:?}"))?;
    let data = payload.clone();
    let _q = router
        .declare_queryable(KEY)
        .callback(move |q| {
            let _ = q.reply(KEY, data.clone()).wait();
        })
        .await
        .map_err(|e| anyhow!("{e:?}"))?;

    let client =
        zenoh::open(config("client", "connect/endpoints")?).await.map_err(|e| anyhow!("{e:?}"))?;

    let t0 = Instant::now();
    let replies = client
        .get(KEY)
        .timeout(std::time::Duration::from_secs(120))
        .await
        .map_err(|e| anyhow!("{e:?}"))?;
    let Ok(reply) = replies.recv_async().await else {
        println!("{mb} MB: NO REPLY — payload rejected or timed out");
        return Ok(());
    };
    let sample = match reply.result() {
        Ok(s) => s,
        Err(e) => {
            println!("{mb} MB: ERROR reply: {e:?}");
            return Ok(());
        }
    };
    let got = sample.payload().to_bytes();
    let dt = t0.elapsed().as_secs_f64();
    let sum: u64 = got.iter().map(|&b| b as u64).sum();

    println!(
        "{:>5} MB | {:6.2}s | {:7.0} MB/s | got {:>5} MB | intact: {}",
        mb,
        dt,
        (got.len() as f64 / 1048576.0) / dt,
        got.len() / 1048576,
        if got.len() == len && sum == expect_sum { "yes" } else { "NO" }
    );
    Ok(())
}
