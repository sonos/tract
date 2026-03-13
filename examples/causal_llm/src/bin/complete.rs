use anyhow::Result;
use causal_llm::{CausalLlmModel, CausalLlmModelConfig};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(about = "Run text completion on a causal LLM")]
struct Args {
    /// Path to tokenizer.json
    #[arg(short, long)]
    tokenizer: String,

    /// Path to NNEF model (.nnef.tgz or directory)
    #[arg(short, long)]
    model: String,

    /// Number of tokens to generate
    #[arg(short, default_value = "20")]
    n: usize,

    /// Force CPU execution (no CUDA or Metal)
    #[arg(long)]
    force_cpu: bool,

    /// Prompts to complete
    prompts: Vec<String>,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    let conf = CausalLlmModelConfig { force_cpu: args.force_cpu };
    let llm = CausalLlmModel::from_paths_and_conf(&args.tokenizer, &args.model, conf)?;

    for prompt in &args.prompts {
        let mut state = llm.spawn()?;
        state.append_text(prompt)?;
        for _ in 0..args.n {
            state.generate_next_token()?;
        }
        let prompt_len = state.encode(prompt, true)?.len();
        let generated = state.decode(&state.seq[prompt_len..], true)?;
        println!("{prompt}{generated}");
    }

    Ok(())
}
