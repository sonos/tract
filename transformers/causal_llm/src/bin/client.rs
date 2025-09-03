use std::io::Write;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};
use clap::Parser;
use reqwest::Client;
use tokenizers::Tokenizer;

mod proto;
use proto::*;

#[derive(Parser, Debug)]
struct Args {
    /// path to some tokenizes.json.file to generate text of the right nuber of token
    #[arg(short, long, required = true)]
    tokenizers: String,

    /// /completions endpoint to bench
    #[structopt(long, default_value = "http://localhost:3000/v1/completions")]
    endpoint: String,

    /// model id
    #[structopt(long, default_value = "meta-llama/Llama-3.1-8B")]
    model: String,

    #[command(subcommand)]
    command: Commands,

    #[structopt(long)]
    api: Option<Api>,
}

#[derive(Parser, Debug, Copy, Clone)]
enum Api {
    Completions,
    Generate,
}

impl FromStr for Api {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "completions" => Ok(Api::Completions),
            "generate" => Ok(Api::Generate),
            _ => bail!("Unknown api name {s}"),
        }
    }
}

#[derive(Parser, Debug)]
enum Commands {
    GenerateBench(GenerateBenchArgs),
    Stress { concurrency: usize, avg_pp: usize, avg_tg: usize },
}

#[derive(Parser, Debug)]
struct GenerateBenchArgs {
    #[arg(value_delimiter = ',')]
    pp: Vec<usize>,
    #[arg(value_delimiter = ',')]
    tg: Vec<usize>,
    #[arg(long)]
    time: Option<f32>,
}

impl Commands {
    async fn run(&self, clients: &Clients) -> Result<()> {
        match self {
            Self::GenerateBench(args) => {
                for tg in &args.tg {
                    print!("\t{tg}");
                }
                println!("");
                for &pp in &args.pp {
                    print!("{pp}");
                    for &tg in &args.tg {
                        // this is wall clock to stop the benching loop
                        let start = Instant::now();
                        // this will measure duration for successul runs  only
                        let mut duration: Duration = Default::default();
                        for ran in 1.. {
                            duration += clients.bench_one_generate(pp, tg).await?;
                            if !args.time.is_some_and(|limit| start.elapsed().as_secs_f32() < limit)
                            {
                                let d = duration.as_secs_f32() / ran as f32;
                                print!("\t{d:.3}");
                                let _ = std::io::stdout().flush();
                                break;
                            }
                        }
                    }
                    println!("");
                }
            }
            Self::Stress { concurrency, avg_pp, avg_tg } => {
                use Ordering::Relaxed;
                tokio_scoped::scope(|scope| {
                    let total_pp = Arc::new(AtomicUsize::default());
                    let total_tg = Arc::new(AtomicUsize::default());
                    let (tpp, ttg) = (total_pp.clone(), total_tg.clone());
                    let report = Duration::from_secs(10);
                    scope.spawn(async move {
                        let mut interval = tokio::time::interval(report);
                        loop {
                            interval.tick().await;
                            eprintln!(
                                "PP:{:.0}/s TG:{:.0}/s",
                                tpp.swap(0, Relaxed) as f32 / report.as_secs_f32(),
                                ttg.swap(0, Relaxed) as f32 / report.as_secs_f32(),
                            );
                        }
                    });
                    for _ in 0..*concurrency {
                        let (tpp, ttg) = (total_pp.clone(), total_tg.clone());
                        scope.spawn(async move {
                            loop {
                                let pp = ((*avg_pp as f32) * (0.8 + 0.4 * rand::random::<f32>()))
                                    as usize;
                                let tg = (((*avg_tg as f32) * (0.8 + 0.4 * rand::random::<f32>()))
                                    as usize)
                                    .max(1);
                                if let Ok(it) = clients.run_one_generate(pp, tg).await {
                                    tpp.fetch_add(it.usage.prompt_tokens, Relaxed);
                                    ttg.fetch_add(it.usage.completion_tokens, Relaxed);
                                }
                            }
                        });
                    }
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct Clients {
    endpoint: String,
    model: String,
    client: Client,
    tokens: Vec<u32>,
    tokenizer: Tokenizer,
    api: Api,
}

impl Clients {
    fn from_args(args: &Args) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(&args.tokenizers)
            .map_err(|e| anyhow!("Failed to initialize tokenzizers: {e:}"))?;
        const BASE_TEXT: &str = include_str!("../lib.rs");
        let tokens: Vec<u32> = tokenizer.encode_fast(&*BASE_TEXT, true).unwrap().get_ids().into();
        let api = args.api.unwrap_or(if args.endpoint.ends_with("generate") {
            Api::Generate
        } else {
            Api::Completions
        });
        Ok(Clients {
            endpoint: args.endpoint.to_string(),
            api,
            model: args.model.clone(),
            client: Client::new(),
            tokens,
            tokenizer,
        })
    }
    fn get_one_prompt(&self, len: usize) -> String {
        assert!(len < self.tokens.len());
        let start: usize = rand::random::<u32>() as usize % (self.tokens.len() - len);
        self.tokenizer.decode(&self.tokens[start..][..len], true).unwrap()
    }

    async fn run_one_generate(&self, pp: usize, tg: usize) -> Result<OpenAICompletionReply> {
        match self.api {
            Api::Completions => self.run_one_generate_openai(pp, tg).await,
            Api::Generate => self.run_one_generate_ollama(pp, tg).await,
        }
    }

    async fn run_one_generate_ollama(&self, pp: usize, tg: usize) -> Result<OpenAICompletionReply> {
        let query = OllamaCompletionQuery {
            prompt: self.get_one_prompt(pp),
            model: self.model.clone(),
            options: OllamaCompletionOptions { num_predict: tg },
        };
        let response = self
            .client
            .post(&self.endpoint)
            .body(serde_json::to_string(&query)?)
            .header("content-type", "application/json")
            .send()
            .await?;
        if response.status().is_success() {
            let it = response.text().await?;
            dbg!(&it);
            Ok(serde_json::from_str(&it)?)
        } else {
            let error = response.text().await.unwrap();
            anyhow::bail!(error)
        }
    }

    async fn run_one_generate_openai(&self, pp: usize, tg: usize) -> Result<OpenAICompletionReply> {
        let query = OpenAICompletionQuery {
            prompt: self.get_one_prompt(pp),
            model: self.model.clone(),
            max_tokens: tg,
            stop: vec![],
        };
        let response = self
            .client
            .post(&self.endpoint)
            .body(serde_json::to_string(&query)?)
            .header("content-type", "application/json")
            .send()
            .await?;
        if response.status().is_success() {
            let it = response.text().await?;
            Ok(serde_json::from_str(&it)?)
        } else {
            let error = response.text().await.unwrap();
            anyhow::bail!(error)
        }
    }

    async fn bench_one_generate(&self, pp: usize, tg: usize) -> Result<Duration> {
        for _attempt in 0..10 {
            let start = Instant::now();
            let it = self.run_one_generate(pp, tg).await?;
            if it.usage.completion_tokens == tg {
                return Ok(start.elapsed());
            };
        }
        bail!("Failed to obtain enough tokens from generate after multiple attempts")
    }
}

#[tokio::main(flavor = "multi_thread", worker_threads = 8)]
async fn main() -> Result<()> {
    let cli = Args::parse();
    let clients = Clients::from_args(&cli)?;

    let start = Instant::now();
    cli.command.run(&clients).await?;

    let total = start.elapsed();
    dbg!(total);
    Ok(())
}
