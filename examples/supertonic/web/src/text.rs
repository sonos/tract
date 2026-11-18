//! Supertonic's text front-end: Unicode normalization, the symbol clean-up of the
//! reference implementation, language tags, and splitting long text into chunks.

use unicode_normalization::UnicodeNormalization;

const MAX_CHUNK: usize = 300;
const MAX_CHUNK_CJK: usize = 120;

const REPLACEMENTS: &[(&str, &str)] = &[
    ("–", "-"),
    ("‑", "-"),
    ("—", "-"),
    ("_", " "),
    ("\u{201C}", "\""),
    ("\u{201D}", "\""),
    ("\u{2018}", "'"),
    ("\u{2019}", "'"),
    ("´", "'"),
    ("`", "'"),
    ("[", " "),
    ("]", " "),
    ("|", " "),
    ("/", " "),
    ("#", " "),
    ("→", " "),
    ("←", " "),
    ("♥", ""),
    ("☆", ""),
    ("♡", ""),
    ("©", ""),
    ("\\", ""),
    ("@", " at "),
    ("e.g.,", "for example, "),
    ("i.e.,", "that is, "),
];

const ABBREVIATIONS: &[&str] = &[
    "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Sr.", "Jr.", "St.", "Ave.", "Rd.", "Blvd.", "Dept.",
    "Inc.", "Ltd.", "Co.", "Corp.", "etc.", "vs.", "i.e.", "e.g.", "Ph.D.",
];

fn is_emoji(c: char) -> bool {
    matches!(c as u32,
        0x1F300..=0x1F6FF | 0x1F700..=0x1FAFF | 0x2600..=0x27BF | 0x1F1E6..=0x1F1FF)
}

/// Normalizes one chunk of text and wraps it in its language tags, as the model was trained.
pub fn prepare(text: &str, lang: &str) -> String {
    let mut text: String = text.nfkd().filter(|c| !is_emoji(*c)).collect();
    for (from, to) in REPLACEMENTS {
        text = text.replace(from, to);
    }
    for punct in [",", ".", "!", "?", ";", ":", "'"] {
        text = text.replace(&format!(" {punct}"), punct);
    }
    for quote in ["\"", "'"] {
        let double = quote.repeat(2);
        while text.contains(&double) {
            text = text.replace(&double, quote);
        }
    }
    let mut text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if let Some(last) = text.chars().last()
        && !".!?;:,'\"\u{201C}\u{201D}\u{2018}\u{2019})]}…。」』】〉》›»".contains(last)
    {
        text.push('.');
    }
    format!("<{lang}>{text}</{lang}>")
}

/// Splits text into chunks the model reads well: paragraphs, then sentences, then clauses,
/// then words, each chunk at most a few hundred characters.
pub fn chunks(text: &str, lang: &str) -> Vec<String> {
    let max = if lang == "ko" || lang == "ja" { MAX_CHUNK_CJK } else { MAX_CHUNK };
    let mut chunks = vec![];
    for paragraph in paragraphs(text) {
        let mut current = String::new();
        for sentence in sentences(&paragraph) {
            let pieces = if sentence.chars().count() > max {
                split_long(&sentence, max)
            } else {
                vec![sentence]
            };
            for piece in pieces {
                if !current.is_empty() && current.chars().count() + 1 + piece.chars().count() > max
                {
                    chunks.push(std::mem::take(&mut current));
                }
                if !current.is_empty() {
                    current.push(' ');
                }
                current.push_str(&piece);
            }
        }
        if !current.is_empty() {
            chunks.push(current);
        }
    }
    chunks
}

fn paragraphs(text: &str) -> Vec<String> {
    let mut paragraphs = vec![String::new()];
    for line in text.lines() {
        if line.trim().is_empty() {
            paragraphs.push(String::new());
        } else {
            let current = paragraphs.last_mut().unwrap();
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(line.trim());
        }
    }
    paragraphs.retain(|p| !p.is_empty());
    paragraphs
}

fn sentences(paragraph: &str) -> Vec<String> {
    let mut sentences = vec![];
    let mut current = String::new();
    let mut chars = paragraph.chars().peekable();
    while let Some(c) = chars.next() {
        current.push(c);
        let boundary = matches!(c, '.' | '!' | '?' | '。' | '！' | '？')
            && chars.peek().is_none_or(|n| n.is_whitespace())
            && !ABBREVIATIONS.iter().any(|a| current.ends_with(a));
        if boundary {
            sentences.push(current.trim().to_string());
            current.clear();
        }
    }
    if !current.trim().is_empty() {
        sentences.push(current.trim().to_string());
    }
    sentences
}

fn split_long(sentence: &str, max: usize) -> Vec<String> {
    let mut pieces = vec![];
    let mut current = String::new();
    for word in sentence.split_whitespace() {
        let fits = current.chars().count() + 1 + word.chars().count() <= max;
        let clause_end = current.ends_with(',') && current.chars().count() > max / 2;
        if !current.is_empty() && (!fits || clause_end) {
            pieces.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    if !current.is_empty() {
        pieces.push(current);
    }
    pieces
}
