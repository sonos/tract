//! Decode-throughput benchmark for the Metal GGML f32-round-trip change.
//! Loads a causal LLM (default = Metal runtime), prefills a prompt, then times
//! N decode steps and prints tokens/sec + the generated text (for a correctness
//! eyeball). Usage: complete_bench <tokenizer.json> <model.nnef.tgz> [n] [--cpu]
use anyhow::Result;
use causal_llm::{CausalLlmModel, CausalLlmModelConfig};
use std::time::Instant;

fn main() -> Result<()> {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();
    let tokenizer = &args[1];
    let model = &args[2];
    let n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    let force_cpu = args.iter().any(|a| a == "--cpu");

    let conf = CausalLlmModelConfig { force_cpu };
    let t0 = Instant::now();
    let llm = CausalLlmModel::from_paths_and_conf(tokenizer, model, conf)?;
    eprintln!("model load: {:?} (force_cpu={force_cpu})", t0.elapsed());

    let prompt = "The capital of France is";
    let mut state = llm.spawn()?;
    state.append_text(prompt)?;
    let prompt_len = state.encode(prompt, true)?.len();

    // Prefill + warm up.
    for _ in 0..3 {
        state.generate_next_token()?;
    }

    let t = Instant::now();
    for _ in 0..n {
        state.generate_next_token()?;
    }
    let dt = t.elapsed();
    let toks_per_s = n as f64 / dt.as_secs_f64();
    let generated = state.decode(&state.seq[prompt_len..], true)?;
    println!(
        "decode: {:.2} tok/s  ({:.3} ms/token over {} tokens)",
        toks_per_s,
        dt.as_secs_f64() * 1e3 / n as f64,
        n
    );
    println!("text: {prompt}{generated}");
    Ok(())
}
