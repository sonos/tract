//! CLIP's byte-level BPE tokenizer, as SD 1.5's text encoder expects it.

use std::collections::HashMap;

use anyhow::{Context, Result};
use regex::Regex;

pub const CONTEXT: usize = 77;
/// Prompt tokens the encoder sees: its positions less the start and end tokens.
pub const PROMPT_TOKENS: usize = CONTEXT - 2;
const START: u32 = 49406;
const END: u32 = 49407;

#[derive(Clone)]
pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    ranks: HashMap<(String, String), usize>,
    byte_chars: [char; 256],
    pattern: Regex,
}

/// GPT-2's reversible map from bytes to printable characters.
fn byte_chars() -> [char; 256] {
    let mut printable: Vec<u32> = ('!' as u32..='~' as u32)
        .chain('¡' as u32..='¬' as u32)
        .chain('®' as u32..='ÿ' as u32)
        .collect();
    let mut chars: Vec<u32> = printable.clone();
    let mut extra = 0;
    for b in 0..256u32 {
        if !printable.contains(&b) {
            printable.push(b);
            chars.push(256 + extra);
            extra += 1;
        }
    }
    let mut table = ['\0'; 256];
    for (b, c) in printable.iter().zip(chars) {
        table[*b as usize] = char::from_u32(c).unwrap();
    }
    table
}

impl Tokenizer {
    /// From the `vocab.json` and `merges.txt` of an HF CLIP tokenizer.
    pub fn new(vocab_json: &str, merges_txt: &str) -> Result<Tokenizer> {
        let vocab: HashMap<String, u32> = serde_json::from_str(vocab_json)?;
        let ranks = merges_txt
            .lines()
            .filter(|l| !l.starts_with("#version") && !l.trim().is_empty())
            .enumerate()
            .map(|(rank, line)| {
                let (a, b) = line.split_once(' ').context("malformed merge")?;
                Ok(((a.to_string(), b.to_string()), rank))
            })
            .collect::<Result<_>>()?;
        let pattern = Regex::new(
            r"(?i)<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
        )?;
        Ok(Tokenizer { vocab, ranks, byte_chars: byte_chars(), pattern })
    }

    fn bpe(&self, word: &str) -> Vec<String> {
        let mut parts: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        if let Some(last) = parts.last_mut() {
            last.push_str("</w>");
        }
        loop {
            let best = parts
                .windows(2)
                .enumerate()
                .filter_map(|(i, w)| self.ranks.get(&(w[0].clone(), w[1].clone())).map(|r| (*r, i)))
                .min();
            let Some((_, i)) = best else { break };
            let pair = (parts[i].clone(), parts[i + 1].clone());
            let mut merged = Vec::with_capacity(parts.len());
            let mut i = 0;
            while i < parts.len() {
                if i + 1 < parts.len() && parts[i] == pair.0 && parts[i + 1] == pair.1 {
                    merged.push(format!("{}{}", pair.0, pair.1));
                    i += 2;
                } else {
                    merged.push(parts[i].clone());
                    i += 1;
                }
            }
            parts = merged;
        }
        parts
    }

    /// Token ids for `text`, framed by the start and end tokens and padded with the end token
    /// to the encoder's 77 positions.
    pub fn encode(&self, text: &str) -> Result<Vec<i64>> {
        let mut ids = vec![START as i64];
        ids.extend(self.tokens(text)?);
        ids.truncate(CONTEXT - 1);
        ids.push(END as i64);
        ids.resize(CONTEXT, END as i64);
        Ok(ids)
    }

    /// The token ids of `text` alone, unframed and untruncated: the prompt fits the encoder
    /// while there are at most [`PROMPT_TOKENS`] of them.
    pub fn tokens(&self, text: &str) -> Result<Vec<i64>> {
        let cleaned = text.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
        let mut ids = vec![];
        for m in self.pattern.find_iter(&cleaned) {
            let word: String = m.as_str().bytes().map(|b| self.byte_chars[b as usize]).collect();
            for piece in self.bpe(&word) {
                let id =
                    self.vocab.get(&piece).with_context(|| format!("{piece:?} not in vocab"))?;
                ids.push(*id as i64);
            }
        }
        Ok(ids)
    }
}
