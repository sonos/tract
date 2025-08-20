use std::sync::Arc;

use axum::extract::State;
use axum::http::{Response, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use axum_macros::debug_handler;
use causal_llm::{CausalLlmModel, CausalLlmStateConfig};
use clap::{arg, command, Parser};
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
}

struct Context {
    llm: Arc<CausalLlmModel>,
}

#[tokio::main]
async fn main() -> TractResult<()> {
    let args = Args::parse();

    let llm = CausalLlmModel::from_paths(args.tokenizers, args.model)?;
    dbg!("model loaded");

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
    let s = tokio::task::spawn_blocking(move || -> Result<OpenAICompletionReply> {
        let mut state = global.llm.spawn()?;
        let mut token = state.process_text(&query.prompt)?;
        let prompt_len = state.seq.len();
        while state.seq.len() - prompt_len < query.max_tokens {
            token = state.process_tokens(&[token])?;
        }
        let generated = state.decode(&state.seq[prompt_len..], true)?;
        Ok(OpenAICompletionReply {
            id: "foo".into(),
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
