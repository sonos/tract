//! Microbenchmark: forward-pass latency as a function of how many tokens are
//! fed in one pass, at a fixed past length. This is the core premise of
//! speculative decoding — verifying K draft tokens should cost about as much as
//! decoding one — and shows where it breaks down (e.g. an all-position LM head
//! over a large vocabulary makes per-pass cost grow with K).

use std::time::Instant;

use anyhow::Result;
use causal_llm::{CausalLlmModel, CausalLlmModelConfig};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(about = "Microbenchmark forward latency vs tokens-per-pass")]
struct Args {
    #[arg(short, long)]
    tokenizer: String,
    #[arg(short, long)]
    model: String,
    /// Past (cached) length to hold fixed while sweeping pass width.
    #[arg(long, default_value = "256")]
    past: usize,
    /// Repetitions per measurement.
    #[arg(long, default_value = "30")]
    reps: usize,
    #[arg(long)]
    force_cpu: bool,
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    let conf = CausalLlmModelConfig { force_cpu: args.force_cpu };
    let model = CausalLlmModel::from_paths_speculative(&args.tokenizer, &args.model, conf)?;

    // Build a fixed past of ~`past` tokens.
    let mut state = model.spawn()?;
    let filler = "the quick brown fox jumps over the lazy dog and then ".repeat(64);
    state.append_text(&filler)?;
    state.seq.truncate(args.past.max(1));
    state.generate_next_token()?; // process the past; sets processed = past

    let widths = [1usize, 2, 3, 4, 6, 8, 12, 16, 24, 32];
    // Warm up.
    for _ in 0..3 {
        state.bench_forward(&[100u32; 4])?;
    }

    println!("\npast = {} tokens, {} reps each", args.past, args.reps);
    println!("{:>6} {:>12} {:>14} {:>16}", "S", "ms/pass", "ms/token", "vs S=1 (per-pass)");
    println!("{}", "-".repeat(54));
    let mut base = 0.0;
    for (i, &s) in widths.iter().enumerate() {
        let tokens = vec![100u32; s];
        let mut samples = Vec::with_capacity(args.reps);
        for _ in 0..args.reps {
            let t = Instant::now();
            state.bench_forward(&tokens)?;
            samples.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let ms = median(samples);
        if i == 0 {
            base = ms;
        }
        println!("{s:>6} {ms:>12.3} {:>14.3} {:>15.2}x", ms / s as f64, ms / base);
    }
    println!(
        "\nIf ms/pass were flat, verifying K drafts would be free; growth with S is \
         the per-pass overhead speculation must beat via accepted tokens."
    );
    Ok(())
}
