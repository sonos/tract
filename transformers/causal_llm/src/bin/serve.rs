use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use axum_macros::debug_handler;
use causal_llm::{CausalLlmModel, CausalLlmStateConfig};
use clap::{Parser, arg, command};
use log::{debug, info};
use tract_nnef::prelude::TractResult;

#[allow(dead_code)]
mod common;
use common::*;

macro_rules! http_ensure {
    ($expr: expr, $msg: expr) => {
        if !$expr {
            return Err(anyhow::anyhow!($msg).into());
        }
    };
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

    /// Disable prefill chunking
    #[arg(long)]
    no_prefill_chunk: bool,

    /// Prefill chunking
    #[arg(long, default_value = "512")]
    prefill_chunk: usize,
}

struct Context {
    args: Args,
    llm: Arc<CausalLlmModel>,
}

#[tokio::main]
async fn main() -> TractResult<()> {
    let args = Args::parse();

    if ::std::env::var("TRACT_SERVE_LOG").is_err() {
        let level = match args.verbose {
            0 => "serve=warn,causal_llm=warn",
            1 => "serve=info,causal_llm=info",
            2 => "serve=debug,causal_llm=debug",
            _ => "serve=trace,causal_llm=trace",
        };
        unsafe {
            std::env::set_var("TRACT_SERVE_LOG", level);
        }
    }

    let env = env_logger::Env::default().filter_or("TRACT_SERVE_LOG", "warn");

    env_logger::Builder::from_env(env).format_timestamp_nanos().init();

    let conf = causal_llm::CausalLlmModelConfig { force_cpu: args.force_cpu };
    let llm = CausalLlmModel::from_paths_and_conf(&args.tokenizers, &args.model, conf)?;
    info!("model loaded");

    let context = Context { llm, args };

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

#[debug_handler]
async fn completions(
    State(global): State<Arc<Context>>,
    Json(query): Json<OpenAICompletionQuery>,
) -> Result<Json<OpenAICompletionReply>> {
    http_ensure!(query.max_tokens > 0, "max_tokens must be at least 1");

    static COUNTER: AtomicUsize = AtomicUsize::new(1);
    let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed).to_string();
    let s = tokio::task::spawn_blocking(move || -> Result<OpenAICompletionReply> {
        let mut state = global.llm.spawn_with_config(CausalLlmStateConfig {
            prompt_chunk_size: Some(global.args.prefill_chunk)
                .filter(|_| !global.args.no_prefill_chunk),
            ..Default::default()
        })?;
        debug!("prompt [{id}] << {}", query.prompt);
        state.append_text(&query.prompt)?;
        let prompt_len = state.seq.len();

        while state.seq.len() - prompt_len < query.max_tokens {
            state.generate_next_token()?;
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
