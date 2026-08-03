//! Chat tokenization for the dashboard: build a ChatML prompt, decode a reply, and
//! report the ids that end an assistant turn.
//!
//! Special tokens are resolved by name from the tokenizer file rather than
//! hardcoded, so a model whose vocabulary numbers them differently still works.

use anyhow::{Context, Result, anyhow};
use tokenizers::Tokenizer;

/// A ChatML tokenizer over a `tokenizer.json`.
///
/// The tokenizer is configured so that control tokens appearing in a message are
/// encoded as ordinary text; the turn structure is assembled from ids here. A
/// message can therefore never close its own turn or forge another role's.
pub struct ChatTokenizer {
    tok: Tokenizer,
    im_start: u32,
    im_end: u32,
    eos: Option<u32>,
    think_open: Option<u32>,
    think_close: Option<u32>,
}

impl ChatTokenizer {
    pub fn from_file(path: &str) -> Result<Self> {
        let mut tok = Tokenizer::from_file(path).map_err(|e| anyhow!("{e}"))?;
        let id = |t: &str| tok.token_to_id(t);
        let im_start =
            id("<|im_start|>").context("tokenizer has no <|im_start|>: not a ChatML model")?;
        let im_end = id("<|im_end|>").context("tokenizer has no <|im_end|>: not a ChatML model")?;
        let (eos, think_open, think_close) = (id("<|endoftext|>"), id("<think>"), id("</think>"));
        tok.set_encode_special_tokens(true);
        Ok(ChatTokenizer { tok, im_start, im_end, eos, think_open, think_close })
    }

    fn text(&self, s: &str) -> Result<Vec<u32>> {
        Ok(self.tok.encode(s, false).map_err(|e| anyhow!("{e}"))?.get_ids().to_vec())
    }

    /// A single user turn, with thinking suppressed so the answer comes back directly.
    ///
    /// Suppression prefills an empty thinking block on the assistant turn rather than
    /// appending a `/no_think` switch to the text: the switch is part of the user's
    /// message, so the model reads it as content and a short prompt is dominated by it.
    pub fn encode_prompt(&self, text: &str) -> Result<Vec<i64>> {
        let mut ids = vec![self.im_start];
        ids.extend(self.text("user\n")?);
        ids.extend(self.text(text)?);
        ids.push(self.im_end);
        ids.extend(self.text("\n")?);
        ids.push(self.im_start);
        ids.extend(self.text("assistant\n")?);
        if let (Some(open), Some(close)) = (self.think_open, self.think_close) {
            ids.push(open);
            ids.extend(self.text("\n\n")?);
            ids.push(close);
            ids.extend(self.text("\n\n")?);
        }
        Ok(ids.into_iter().map(|i| i as i64).collect())
    }

    /// Ids that end an assistant turn. The coordinator stops decoding on these
    /// instead of running to `max_tokens`.
    pub fn stop_ids(&self) -> Vec<i64> {
        let mut v = vec![self.im_end as i64];
        v.extend(self.eos.map(|e| e as i64));
        v
    }

    /// The assistant's answer: everything up to the turn end, with any thinking
    /// block dropped. While the model is still inside `<think>` there is no answer
    /// yet, so this yields an empty string rather than leaking the block.
    pub fn decode_reply(&self, ids: &[i64]) -> Result<String> {
        let mut ids: Vec<u32> = ids.iter().filter_map(|&i| u32::try_from(i).ok()).collect();
        if let Some(p) = ids.iter().position(|&i| i == self.im_end) {
            ids.truncate(p);
        }
        if let Some(p) = self.think_close.and_then(|t| ids.iter().position(|&i| i == t)) {
            ids.drain(..=p);
        } else if self.think_open.is_some_and(|t| ids.contains(&t)) {
            return Ok(String::new());
        }
        Ok(self.tok.decode(&ids, true).map_err(|e| anyhow!("{e}"))?.trim().to_string())
    }
}
