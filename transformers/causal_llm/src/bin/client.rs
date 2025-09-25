use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::primitives::Blob;
use clap::Parser;
use reqwest::Client;
use tokenizers::Tokenizer;
use tokio::sync::broadcast::Receiver;

#[allow(dead_code)]
mod common;
use common::*;

#[derive(Parser, Debug)]
struct Args {
    /// tokenizer to use for token counts
    #[arg(long, short, required(true))]
    tokenizers: String,

    /// /completions endpoint to bench
    #[arg(long, default_value = "http://localhost:3000/v1/completions")]
    endpoint: String,

    /// model id
    #[arg(long, default_value = "meta-llama/Llama-3.1-8B")]
    model: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Clone)]
enum Api {
    OpenAICompletions(Client, String),
    OllamaGenerate(Client, String),
    AWSBedrockInvokeModel(aws_sdk_bedrockruntime::Client),
}

struct GenericCompletion {
    text: String,
    generated_tokens: usize,
    prompt_tokens: usize,
}

impl Api {
    async fn new(endpoint: &str) -> Api {
        if endpoint == "awsbedrock" {
            let config = aws_config::defaults(BehaviorVersion::latest()).load().await;
            let client = aws_sdk_bedrockruntime::Client::new(&config);
            Api::AWSBedrockInvokeModel(client)
        } else if endpoint.ends_with("generate") {
            Api::OllamaGenerate(Client::new(), endpoint.to_string())
        } else {
            Api::OpenAICompletions(Client::new(), endpoint.to_string())
        }
    }

    async fn generate(
        &self,
        model: &str,
        prompt: impl Into<String>,
        max_tokens: usize,
    ) -> Result<GenericCompletion> {
        match &self {
            Api::OpenAICompletions(client, endpoint) => {
                let query = OpenAICompletionQuery {
                    prompt: prompt.into(),
                    model: model.to_string(),
                    max_tokens,
                    stop: vec![],
                };
                let response = client
                    .post(endpoint)
                    .body(serde_json::to_string(&query)?)
                    .header("content-type", "application/json")
                    .send()
                    .await?;
                if response.status().is_success() {
                    let it: OpenAICompletionReply = serde_json::from_str(&response.text().await?)?;
                    Ok(GenericCompletion {
                        text: it.choices[0].text.clone(),
                        generated_tokens: it.usage.completion_tokens,
                        prompt_tokens: it.usage.prompt_tokens,
                    })
                } else {
                    let error = response.text().await.unwrap();
                    anyhow::bail!(error)
                }
            }
            Api::OllamaGenerate(client, endpoint) => {
                let query = OllamaCompletionQuery {
                    prompt: prompt.into(),
                    model: model.to_string(),
                    options: OllamaCompletionOptions { num_predict: max_tokens },
                };

                let response = client
                    .post(endpoint)
                    .body(serde_json::to_string(&query)?)
                    .header("content-type", "application/json")
                    .send()
                    .await?;

                if response.status().is_success() {
                    todo!()
                    // let it: Oll = serde_json::from_str(response.text().awai?)?;
                    // Ok(GenericCompletion {
                    //     text: it.choices[0].text,
                    //     generated_tokens: it.usage.completion_tokens,
                    //     prompt_tokens: it.usage.prompt_tokens,
                    // })
                } else {
                    let error = response.text().await.unwrap();
                    anyhow::bail!(error)
                }
            }
            Api::AWSBedrockInvokeModel(client) => {
                let stop: [String; 0] = [];
                let query = serde_json::json!({ "prompt" : prompt.into(), "max_gen_len": max_tokens, "stop":stop}).to_string();
                let resp = client
                    .invoke_model()
                    .model_id(model)
                    .content_type("application/json")
                    .accept("application/json")
                    .body(Blob::new(query.into_bytes()))
                    .send()
                    .await?;
                let raw = String::from_utf8(resp.body.into_inner())?;
                let v: serde_json::Value = serde_json::from_str(&raw)?;
                Ok(GenericCompletion {
                    text: v.get("generation").unwrap().as_str().unwrap().to_string(),
                    generated_tokens: v.get("generation_token_count").unwrap().as_i64().unwrap()
                        as usize,
                    prompt_tokens: v.get("prompt_token_count").unwrap().as_i64().unwrap() as usize,
                })
            }
        }
    }
}

#[derive(Parser, Debug)]
enum Commands {
    GenerateBench(GenerateBenchArgs),
    Stress(StressArgs),
    Scalability(ScalabilityArgs),
    Complete(CompleteArgs),
    Prompts(PromptsArgs),
}

#[derive(Parser, Debug)]
struct GenerateBenchArgs {
    #[structopt(value_delimiter = ',')]
    pp: Vec<usize>,
    #[structopt(value_delimiter = ',')]
    tg: Vec<usize>,
    #[structopt(long)]
    time: Option<f32>,
}

impl GenerateBenchArgs {
    async fn handle(&self, clients: &Clients) -> Result<()> {
        for tg in &self.tg {
            print!("\t{tg}");
        }
        println!();
        for &pp in &self.pp {
            print!("{pp}");
            for &tg in &self.tg {
                let dur = self.run_one(clients, pp, tg).await?;
                print!("\t{:.3}", dur.as_secs_f32());
                let _ = std::io::stdout().flush();
            }
            println!();
        }
        Ok(())
    }

    async fn run_one(&self, clients: &Clients, pp: usize, tg: usize) -> Result<Duration> {
        // this is wall clock to stop the benching loop
        let start = Instant::now();
        // this will measure duration for successul runs  only
        let mut duration: Duration = Default::default();
        for ran in 1.. {
            duration += clients.bench_one_generate(pp, tg).await?;
            if !self.time.is_some_and(|limit| start.elapsed().as_secs_f32() < limit) {
                return Ok(duration / ran);
            }
        }
        unreachable!()
    }
}

#[derive(Parser, Debug)]
struct StressArgs {
    concurrency: usize,
    avg_pp: usize,
    avg_tg: usize,
    /// do not report aggregated performance
    #[arg(long)]
    quiet: bool,
}

impl StressArgs {
    async fn handle(&self, clients: &Clients) -> Result<()> {
        self.stress(clients, None).await
    }

    async fn stress(&self, clients: &Clients, keepalive: Option<Receiver<()>>) -> Result<()> {
        use Ordering::Relaxed;
        let total_pp = Arc::new(AtomicUsize::default());
        let total_tg = Arc::new(AtomicUsize::default());
        let report = Duration::from_secs(10);
        tokio_scoped::scope(|scope| {
            if !self.quiet {
                scope.spawn(async {
                    let mut interval = tokio::time::interval(report);
                    loop {
                        interval.tick().await;
                        eprintln!(
                            "PP:{:.0}/s TG:{:.0}/s",
                            total_pp.swap(0, Relaxed) as f32 / report.as_secs_f32(),
                            total_tg.swap(0, Relaxed) as f32 / report.as_secs_f32(),
                        );
                        if let Some(keepalive) = &keepalive {
                            if keepalive.is_closed() {
                                break;
                            }
                        }
                    }
                });
            }
            for _ in 0..self.concurrency {
                scope.spawn(async {
                    loop {
                        let pp =
                            ((self.avg_pp as f32) * (0.8 + 0.4 * rand::random::<f32>())) as usize;
                        let tg = (((self.avg_tg as f32) * (0.8 + 0.4 * rand::random::<f32>()))
                            as usize)
                            .max(1);
                        if let Ok(it) = clients.run_one_generate(pp, tg).await {
                            total_pp.fetch_add(it.prompt_tokens, Relaxed);
                            total_tg.fetch_add(it.generated_tokens, Relaxed);
                        }
                        if let Some(keepalive) = &keepalive {
                            if keepalive.is_closed() {
                                break;
                            }
                        }
                    }
                });
            }
        });
        Ok(())
    }
}

fn clear_terminal_line() {
    eprint!("\x1b[2K\r");
}

#[derive(Parser, Debug)]
struct ScalabilityArgs {
    avg_pp: usize,
    avg_tg: usize,
    #[arg(value_delimiter = ',')]
    concurrency: Vec<usize>,
}

impl ScalabilityArgs {
    async fn handle(&self, clients: &Clients) -> Result<()> {
        eprint!("Server and model warmup...");
        let bench =
            GenerateBenchArgs { pp: vec![self.avg_pp], tg: vec![self.avg_tg], time: Some(10.0) };
        let _dur = bench.run_one(clients, self.avg_pp, self.avg_tg).await?;
        clear_terminal_line();
        for &concurrency in &self.concurrency {
            tokio_scoped::scope(|scope| {
                eprint!("Starting stress loading with {concurrency} and warming up...");
                let (tx, rx) = tokio::sync::broadcast::channel::<()>(1);
                let stress = StressArgs {
                    quiet: true,
                    concurrency,
                    avg_pp: self.avg_pp,
                    avg_tg: self.avg_tg,
                };
                scope.spawn(async move {
                    let _ = stress.stress(clients, Some(rx)).await;
                });
                scope.spawn(async move {
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    clear_terminal_line();
                    eprint!("Running actual bench");
                    let bench = GenerateBenchArgs {
                        pp: vec![self.avg_pp],
                        tg: vec![self.avg_tg],
                        time: Some(10.0),
                    };
                    let dur = bench.run_one(clients, self.avg_pp, self.avg_tg).await.unwrap();
                    clear_terminal_line();
                    println!("{concurrency} {:.3}", dur.as_secs_f32());
                    std::mem::drop(tx);
                });
            })
        }
        Ok(())
    }
}

#[derive(Parser, Debug)]
struct CompleteArgs {
    prompt: String,
    #[arg(short('n'), default_value = "50")]
    max_tokens: usize,
}

impl CompleteArgs {
    async fn handle(&self, clients: &Clients) -> Result<()> {
        let reply = clients.complete(&self.prompt, self.max_tokens).await?;
        println!("{}", reply.text);
        eprintln!("prompt:{:?} generated:{}", reply.prompt_tokens, reply.generated_tokens);
        Ok(())
    }
}

#[derive(Parser, Debug)]
struct PromptsArgs {
    #[arg(short, long, default_value = "50")]
    len: usize,

    #[arg(short, long, default_value = "50")]
    count: usize,
}

impl PromptsArgs {
    async fn handle(&self, clients: &Clients) -> Result<()> {
        for _ in 0..self.count {
            let prompt = clients.get_one_prompt(self.len);
            println!("{}", prompt.replace("\n", " "));
        }
        Ok(())
    }
}

impl Commands {
    async fn run(&self, clients: &Clients) -> Result<()> {
        match self {
            Self::GenerateBench(args) => args.handle(clients).await?,
            Self::Stress(args) => args.handle(clients).await?,
            Self::Scalability(args) => args.handle(clients).await?,
            Self::Complete(args) => args.handle(clients).await?,
            Self::Prompts(args) => args.handle(clients).await?,
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct Clients {
    model: String,
    tokens: Vec<u32>,
    tokenizer: Tokenizer,
    api: Api,
}

impl Clients {
    async fn from_args(args: &Args) -> Self {
        let tokenizer = Tokenizer::from_file(&args.tokenizers).unwrap();
        const BASE_TEXT: &str = include_str!("../lib.rs");
        let tokens: Vec<u32> = tokenizer.encode_fast(BASE_TEXT, true).unwrap().get_ids().into();
        let api = Api::new(&args.endpoint).await;
        Clients { api, model: args.model.clone(), tokens, tokenizer }
    }
    fn get_one_prompt(&self, len: usize) -> String {
        assert!(len < self.tokens.len());
        let start: usize = rand::random::<u32>() as usize % (self.tokens.len() - len);
        self.tokenizer.decode(&self.tokens[start..][..len.saturating_sub(1)], true).unwrap()
    }

    async fn run_one_generate(&self, pp: usize, tg: usize) -> Result<GenericCompletion> {
        self.api.generate(&self.model, &self.get_one_prompt(pp), tg).await
    }

    async fn complete(
        &self,
        prompt: impl Into<String>,
        max_tokens: usize,
    ) -> Result<GenericCompletion> {
        self.api.generate(&self.model, prompt.into(), max_tokens).await
    }

    async fn bench_one_generate(&self, pp: usize, tg: usize) -> Result<Duration> {
        for _attempt in 0..10 {
            let start = Instant::now();
            let it = self.run_one_generate(pp, tg).await?;
            if it.generated_tokens == tg {
                return Ok(start.elapsed());
            };
        }
        bail!("Failed to obtain enough tokens from generate after multiple attempts")
    }
}

#[tokio::main(flavor = "multi_thread", worker_threads = 8)]
async fn main() -> anyhow::Result<()> {
    let cli = Args::parse();
    let clients = Clients::from_args(&cli).await;

    let start = Instant::now();
    cli.command.run(&clients).await?;

    let total = start.elapsed();
    dbg!(total);
    Ok(())
}
