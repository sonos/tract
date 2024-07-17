use itertools::Itertools;
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::sync::{Arc, Mutex};
use string_interner::DefaultStringInterner;
use string_interner::Symbol as _;

use super::TDim;

#[derive(Clone, Default)]
pub struct SymbolScope(pub Arc<Mutex<SymbolScopeData>>);

impl PartialEq for SymbolScope {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for SymbolScope {}

#[derive(Default)]
pub struct SymbolScopeData {
    table: DefaultStringInterner,
    pub manual_rules: HashMap<TDim, TDim>,
}

impl SymbolScope {
    pub fn get(&self, name: &str) -> Option<Symbol> {
        let locked = self.0.lock().unwrap();
        locked.table.get(name).map(|sym| Symbol(self.clone(), sym))
    }

    pub fn sym(&self, name: &str) -> Symbol {
        let mut locked = self.0.lock().unwrap();
        let sym = locked.table.get_or_intern(name);
        Symbol(self.clone(), sym)
    }

    pub fn new_with_prefix(&self, prefix: &str) -> Symbol {
        let mut locked = self.0.lock().unwrap();
        let sym = if locked.table.get(prefix).is_none() {
            locked.table.get_or_intern(prefix)
        } else {
            let mut i = 0;
            loop {
                let s = format!("{prefix}_{i}");
                if locked.table.get(&s).is_none() {
                    break locked.table.get_or_intern(s);
                }
                i += 1;
            }
        };
        Symbol(self.clone(), sym)
    }

    pub fn resolving<R>(&self, sym: &Symbol, f: impl FnOnce(&str) -> R) -> Option<R> {
        self.0.lock().unwrap().table.resolve(sym.1).map(f)
    }

    pub fn add_rule(&self, from: TDim, to: TDim) {
        self.0.lock().unwrap().manual_rules.insert(from, to);
    }

    pub(crate) fn apply_rules(&self, t: &TDim) -> Option<TDim> {
        self.0.lock().unwrap().manual_rules.get(t).cloned()
    }
}

impl fmt::Debug for SymbolScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let locked = self.0.lock().unwrap();
        write!(f, "{}", locked.table.into_iter().map(|(_, s)| s).join(" "))
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct Symbol(SymbolScope, string_interner::DefaultSymbol);

impl Symbol {
    pub fn scope(&self) -> &SymbolScope {
        &self.0
    }
}

impl PartialOrd for Symbol {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Symbol {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.1.cmp(&other.1)
    }
}

impl std::hash::Hash for Symbol {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.1.hash(state)
    }
}

impl std::fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0
            .resolving(self, |s| write!(f, "{s}"))
            .unwrap_or_else(|| write!(f, "<Sym{}>", self.1.to_usize()))
    }
}

impl fmt::Debug for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&self, f)
    }
}

#[derive(Clone, Debug, Default)]
pub struct SymbolValues {
    values: HashMap<Symbol, i64>,
}

impl SymbolValues {
    pub fn with(mut self, s: &Symbol, v: i64) -> Self {
        self.set(s, v);
        self
    }

    pub fn set(&mut self, s: &Symbol, v: i64) {
        self.values.insert(s.clone(), v);
    }

    pub fn get(&self, s: &Symbol) -> Option<i64> {
        self.values.get(s).copied()
    }
}
