use itertools::Itertools;
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::sync::{Arc, Mutex};
use string_interner::DefaultStringInterner;
use string_interner::Symbol as _;

#[derive(Clone, Default)]
pub struct SymbolTable(pub Arc<Mutex<DefaultStringInterner>>);

impl SymbolTable {
    pub fn get(&self, name: &str) -> Option<Symbol> {
        let table = self.0.lock().unwrap();
        table.get(name).map(|sym| Symbol(Arc::clone(&self.0), sym))
    }

    pub fn sym(&self, name: &str) -> Symbol {
        let mut table = self.0.lock().unwrap();
        let sym = table.get_or_intern(name);
        Symbol(Arc::clone(&self.0), sym)
    }

    pub fn new_with_prefix(&self, prefix: &str) -> Symbol {
        let mut table = self.0.lock().unwrap();
        let sym = if table.get(prefix).is_none() {
            table.get_or_intern(prefix)
        } else {
            let mut i = 0;
            loop {
                let s = format!("{prefix}_{i}");
                if table.get(&s).is_none() {
                    break table.get_or_intern(s);
                }
                i += 1;
            }
        };
        Symbol(Arc::clone(&self.0), sym)
    }
}

impl fmt::Debug for SymbolTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let table = self.0.lock().unwrap();
        write!(f, "{}", (&table).into_iter().map(|(_, s)| s).join(" "))
    }
}

#[derive(Clone)]
pub struct Symbol(Arc<Mutex<DefaultStringInterner>>, string_interner::DefaultSymbol);

impl PartialEq for Symbol {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0) && self.1 == other.1
    }
}

impl Eq for Symbol {}

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
        if let Ok(table) = self.0.lock() {
            if let Some(s) = table.resolve(self.1) {
                return write!(f, "{s}");
            }
        }
        write!(f, "<Sym{}>", self.1.to_usize())
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
