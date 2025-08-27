use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use axum_macros::debug_handler;
use causal_llm::CausalLlmModel;
use clap::{arg, command, Parser};
use log::{debug, info};
use serde::{Deserialize, Serialize};
use tract_nnef::prelude::TractResult;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// path to tokenizes.json.file
    #[arg(short, long, required = true)]
    tokenizers: String,

    /// path to nnef neural net model
    #[arg(short, long, required = true)]
    model: String,

    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Force execution on CPU (no cuda or metal)
    #[arg(long)]
    force_cpu: bool,
}

struct Context {
    llm: Arc<CausalLlmModel>,
}

#[tokio::main]
async fn main() -> TractResult<()> {
    let args = Args::parse();

    if ::std::env::var("TRACT_LOG").is_err() {
        let level = match args.verbose {
            0 => "serve=warn",
            1 => "serve=info,tract=warn",
            2 => "serve=debug,tract=info",
            _ => "serve=trace,tract=debug",
        };
        unsafe {
            std::env::set_var("TRACT_LOG", level);
        }
    }

    let env = env_logger::Env::default().filter_or("TRACT_LOG", "warn");

    env_logger::Builder::from_env(env).format_timestamp_nanos().init();

    let conf = causal_llm::CausalLlmModelConfig { force_cpu: args.force_cpu };
    let llm = CausalLlmModel::from_paths_and_conf(args.tokenizers, args.model, conf)?;
    info!("model loaded");

    let context = Context { llm };

    let app = Router::new()
        .route("/", get(root))
        .route("/v1/completions", post(completions))
        .with_state(Arc::new(context));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await?;
    Ok(())
}

async fn root() -> &'static str {
    "tract mini llm completions server\n"
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct OpenAICompletionQuery {
    prompt: String,
    model: String,
    max_tokens: usize,
    stop: Vec<String>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct OpenAICompletionReply {
    id: String,
    choices: Vec<OpenAICompletionReplyChoice>,
    usage: OpenAICompletionReplyUsage,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct OpenAICompletionReplyChoice {
    text: String,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct OpenAICompletionReplyUsage {
    prompt_tokens: usize,
    total_tokens: usize,
    completion_tokens: usize,
}

#[debug_handler]
async fn completions(
    State(global): State<Arc<Context>>,
    Json(query): Json<OpenAICompletionQuery>,
) -> Result<Json<OpenAICompletionReply>> {
    static COUNTER: AtomicUsize = AtomicUsize::new(1);
    let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed).to_string();
    let s = tokio::task::spawn_blocking(move || -> Result<OpenAICompletionReply> {
        let mut state = global.llm.spawn()?;
        debug!("prompt [{id}] << {}", query.prompt);
        let mut token = state.process_text(&query.prompt)?;
        let prompt_len = state.seq.len();
        while state.seq.len() - prompt_len < query.max_tokens {
            token = state.process_tokens(&[token])?;
        }
        let generated = state.decode(&state.seq[prompt_len..], true)?;
        debug!("gen   [{id}] >> {}", generated);
        Ok(OpenAICompletionReply {
            id,
            choices: vec![OpenAICompletionReplyChoice { text: generated }],
            usage: OpenAICompletionReplyUsage {
                prompt_tokens: prompt_len,
                total_tokens: state.seq.len(),
                completion_tokens: state.seq.len() - prompt_len,
            },
        })
    })
    .await??;
    Ok(Json(s))
}

type Result<A> = std::result::Result<A, AppError>;
struct AppError(anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("Something went wrong: {}", self.0))
            .into_response()
    }
}

impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}
