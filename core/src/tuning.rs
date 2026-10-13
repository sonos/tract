//! Per-device tuning table consulted by codegen.
//!
//! An optimisable op computes a tuning key from its concrete facts (see
//! [`crate::ops::einsum::kernel_selection::tuning_key`]) and asks the table
//! whether a choice was recorded for it. A hit replaces the heuristic pick; a
//! miss, or no table at all, leaves codegen exactly as it is.
//!
//! The table is a process-wide default ([`set_default_tuning`]) that a thread
//! can override for a scope ([`with_tuning`]). Entries are plain text, one per
//! line: `<key> <kernel name> <packing index>`; `#` starts a comment.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::RwLock;

use crate::internal::*;

/// A recorded choice for one optimisable op.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Choice {
    pub kernel: String,
    pub packing: usize,
}

/// Tuning key to recorded choice.
#[derive(Clone, Debug, Default)]
pub struct Tuning {
    pub entries: HashMap<String, Choice>,
}

impl Tuning {
    pub fn parse(text: &str) -> TractResult<Tuning> {
        let mut entries = HashMap::new();
        for (ix, line) in text.lines().enumerate() {
            let line = line.split('#').next().unwrap().trim();
            if line.is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split_whitespace().collect();
            let [key, kernel, packing] = fields[..] else {
                bail!("Tuning line {}: expected `<key> <kernel> <packing>`, got {line:?}", ix + 1);
            };
            let packing = packing
                .parse()
                .with_context(|| format!("Tuning line {}: bad packing index", ix + 1))?;
            entries.insert(key.to_string(), Choice { kernel: kernel.to_string(), packing });
        }
        Ok(Tuning { entries })
    }

    pub fn load(path: impl AsRef<Path>) -> TractResult<Tuning> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("Reading tuning file {}", path.display()))?;
        Tuning::parse(&text)
    }

    pub fn get(&self, key: &str) -> Option<&Choice> {
        self.entries.get(key)
    }
}

/// One optimisable op as codegen saw it: its tuning key (`None` while a dim is
/// symbolic) and its candidates, kernel name and packing index, the heuristic
/// pick first.
#[derive(Clone, Debug)]
pub struct CandidateReport {
    pub node: String,
    pub key: Option<String>,
    pub candidates: Vec<(String, usize)>,
}

static DEFAULT_TUNING: RwLock<Option<Arc<Tuning>>> = RwLock::new(None);

thread_local! {
    static TLS_TUNING: RefCell<Option<Arc<Tuning>>> = const { RefCell::new(None) };
    static TLS_RECORDER: RefCell<Option<Vec<CandidateReport>>> = const { RefCell::new(None) };
}

/// Run `f` and collect a [`CandidateReport`] for each optimisable op codegen
/// meets on this thread meanwhile.
pub fn with_recorder<R>(f: impl FnOnce() -> R) -> (R, Vec<CandidateReport>) {
    let previous = TLS_RECORDER.replace(Some(vec![]));
    let result = f();
    let reports = TLS_RECORDER.replace(previous).unwrap_or_default();
    (result, reports)
}

/// Whether a [`with_recorder`] scope is open on this thread.
pub fn recording() -> bool {
    TLS_RECORDER.with_borrow(|r| r.is_some())
}

/// Add `report` to the open [`with_recorder`] scope, if any.
pub fn record(report: CandidateReport) {
    TLS_RECORDER.with_borrow_mut(|r| {
        if let Some(r) = r {
            r.push(report)
        }
    });
}

/// Install `tuning` as the process-wide default (`None` removes it).
pub fn set_default_tuning(tuning: Option<Tuning>) {
    *DEFAULT_TUNING.write().unwrap() = tuning.map(Arc::new);
}

/// Run `f` with `tuning` in effect on this thread, whatever the default.
pub fn with_tuning<R>(tuning: Tuning, f: impl FnOnce() -> R) -> R {
    let previous = TLS_TUNING.replace(Some(Arc::new(tuning)));
    let result = f();
    TLS_TUNING.set(previous);
    result
}

/// Whether any tuning table is in effect on this thread.
pub fn active() -> bool {
    TLS_TUNING.with_borrow(|t| t.is_some()) || DEFAULT_TUNING.read().unwrap().is_some()
}

/// The choice recorded for `key` by the table in effect, if any.
pub fn lookup(key: &str) -> Option<Choice> {
    if let Some(choice) = TLS_TUNING.with_borrow(|t| t.as_ref().map(|t| t.get(key).cloned())) {
        return choice;
    }
    DEFAULT_TUNING.read().unwrap().as_ref().and_then(|t| t.get(key).cloned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_entries_and_comments() {
        let t = Tuning::parse("# header\nk1 kern_a 0\n\nk2 kern_b 1 # trailing\n").unwrap();
        assert_eq!(t.get("k1"), Some(&Choice { kernel: "kern_a".into(), packing: 0 }));
        assert_eq!(t.get("k2"), Some(&Choice { kernel: "kern_b".into(), packing: 1 }));
        assert!(Tuning::parse("k1 kern_a").is_err());
    }

    #[test]
    fn scoped_table_overrides_default_on_this_thread_only() {
        let t = Tuning::parse("scoped_key kern_a 0").unwrap();
        assert!(lookup("scoped_key").is_none());
        let inside = with_tuning(t, || lookup("scoped_key"));
        assert_eq!(inside.map(|c| c.kernel), Some("kern_a".into()));
        assert!(lookup("scoped_key").is_none());
    }
}
