use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OpenAICompletionQuery {
    pub prompt: String,
    pub model: String,
    pub max_tokens: usize,
    pub temperature: Option<f32>,
    pub stop: Vec<String>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OpenAICompletionReply {
    pub id: String,
    pub choices: Vec<OpenAICompletionReplyChoice>,
    pub usage: OpenAICompletionReplyUsage,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OpenAICompletionReplyChoice {
    pub text: String,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OpenAICompletionReplyUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
    pub completion_tokens: usize,
}

#[derive(Serialize, Debug, Clone)]
pub struct OllamaCompletionQuery {
    pub prompt: String,
    pub model: String,
    pub options: OllamaCompletionOptions,
}

#[derive(Serialize, Debug, Clone)]
pub struct OllamaCompletionOptions {
    pub num_predict: usize,
}
