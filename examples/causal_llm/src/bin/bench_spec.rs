//! End-to-end speculative-decoding benchmark: decode throughput of plain greedy
//! vs n-gram speculative decoding across prompt types and draft lengths, with a
//! lossless-output check on every run.

use std::time::Instant;

use anyhow::Result;
use causal_llm::{CausalLlmModel, CausalLlmModelConfig, NgramDrafter};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(about = "Benchmark speculative decoding vs greedy decoding")]
struct Args {
    #[arg(short, long)]
    tokenizer: String,
    #[arg(short, long)]
    model: String,
    /// Decode tokens to time per run (after an untimed prefill).
    #[arg(short, default_value = "128")]
    n: usize,
    /// Draft lengths to sweep.
    #[arg(long, value_delimiter = ',', default_value = "2,4,8")]
    k: Vec<usize>,
    #[arg(long)]
    force_cpu: bool,
}

struct Prompt {
    name: &'static str,
    text: &'static str,
}

const PROMPTS: &[Prompt] = &[
    Prompt {
        name: "repetitive",
        text: "The capital of France is Paris. The capital of Germany is Berlin. \
                The capital of Italy is Rome. The capital of Spain is Madrid. \
                The capital of Portugal is Lisbon. The capital of",
    },
    Prompt {
        name: "code",
        text: "Here is a Rust function and its repeated use:\n\
                fn add(a: i32, b: i32) -> i32 { a + b }\n\
                let x = add(1, 2);\n\
                let y = add(3, 4);\n\
                let z = add(5, 6);\n\
                let w = add(",
    },
    Prompt {
        name: "prose",
        text: "In a quiet village nestled between two hills, an old clockmaker \
                spent his days repairing watches and telling stories to the children who",
    },
];

/// Decode `n` tokens with plain greedy after an untimed prefill; returns
/// (tokens, decode_seconds).
fn run_greedy(
    model: &std::sync::Arc<CausalLlmModel>,
    prompt: &str,
    n: usize,
) -> Result<(Vec<u32>, f64)> {
    let mut s = model.spawn()?;
    s.append_text(prompt)?;
    s.generate_next_token()?; // prefill + first token (untimed)
    let start = Instant::now();
    for _ in 0..n {
        s.generate_next_token()?;
    }
    let secs = start.elapsed().as_secs_f64();
    Ok((s.seq, secs))
}

/// Decode `n` tokens with n-gram speculative decoding after an untimed prefill;
/// returns (tokens, decode_seconds, proposed, accepted, steps).
fn run_spec(
    model: &std::sync::Arc<CausalLlmModel>,
    prompt: &str,
    n: usize,
    k: usize,
) -> Result<(Vec<u32>, f64, usize, usize, usize)> {
    let mut s = model.spawn()?;
    s.append_text(prompt)?;
    s.generate_next_token()?; // prefill + first token (untimed)
    let target = s.seq.len() + n;
    let mut drafter = NgramDrafter::default();
    let (mut proposed, mut accepted, mut steps) = (0, 0, 0);
    let start = Instant::now();
    while s.seq.len() < target {
        let st = s.generate_speculative(&mut drafter, k)?;
        proposed += st.proposed;
        accepted += st.accepted;
        steps += 1;
    }
    let secs = start.elapsed().as_secs_f64();
    Ok((s.seq, secs, proposed, accepted, steps))
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    let conf = CausalLlmModelConfig { force_cpu: args.force_cpu };
    let model = CausalLlmModel::from_paths_speculative(&args.tokenizer, &args.model, conf)?;

    // Warm up (kernels / allocations) so the first timed run is not penalised.
    run_greedy(&model, "Warm up the runtime once.", 8)?;

    println!(
        "\n{:<11} {:>8} {:>10} {:>8} {:>9} {:>10} {:>8}",
        "prompt", "mode", "tok/s", "accept%", "tok/step", "speedup", "lossless"
    );
    println!("{}", "-".repeat(70));

    for p in PROMPTS {
        let (greedy_seq, greedy_secs) = run_greedy(&model, p.text, args.n)?;
        let greedy_tps = args.n as f64 / greedy_secs;
        println!(
            "{:<11} {:>8} {:>10.1} {:>8} {:>9} {:>10} {:>8}",
            p.name, "greedy", greedy_tps, "-", "-", "1.00x", "-"
        );

        for &k in &args.k {
            let (spec_seq, secs, proposed, accepted, steps) = run_spec(&model, p.text, args.n, k)?;
            let tps = args.n as f64 / secs;
            let accept_pct = 100.0 * accepted as f64 / proposed.max(1) as f64;
            let tok_step = (accepted + steps) as f64 / steps as f64;
            let lossless = spec_seq[..greedy_seq.len()] == greedy_seq[..];
            println!(
                "{:<11} {:>8} {:>10.1} {:>7.0}% {:>9.2} {:>9.2}x {:>8}",
                "",
                format!("spec k={k}"),
                tps,
                accept_pct,
                tok_step,
                tps / greedy_tps,
                if lossless { "yes" } else { "NO!" }
            );
        }
        println!();
    }
    Ok(())
}
