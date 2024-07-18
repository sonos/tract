use itertools::Itertools;
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::sync::{Arc, Mutex};
use string_interner::DefaultStringInterner;
use string_interner::Symbol as _;

use crate::TractResult;

use super::parse::parse_inequality;
use super::{parse_tdim, TDim};

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
    pub inequalities: Vec<Inequality>,
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
        match self.0.try_lock() {
            Ok(lock) => lock.table.resolve(sym.1).map(f),
            Err(_) => None,
        }
    }

    pub fn parse_tdim(&self, input: impl AsRef<str>) -> TractResult<TDim> {
        parse_tdim(self, input.as_ref())
    }

    pub fn parse_inequality(&self, input: impl AsRef<str>) -> TractResult<Inequality> {
        parse_inequality(self, input.as_ref())
    }

    pub fn add_inequality(&self, ineq: Inequality) {
        self.0.lock().unwrap().inequalities.push(ineq)
    }

    pub fn prove_positive(&self, t: &TDim) -> bool {
        if let TDim::Val(v) = t {
            return *v >= 0;
        }
        let ineqs = self.0.lock().unwrap().inequalities.clone();
        let positives = ineqs.iter().map(|i| i.as_known_positive()).collect_vec();
        let mut visited = vec![];
        let mut todo = vec![t.clone()];
        while let Some(t) = todo.pop() {
            if t.to_i64().is_ok_and(|i| i >= 0) {
                return true;
            }
            let syms = t.symbols();
            for s in syms {
                let me = t.guess_slope(&s);
                for pos in &positives {
                    if pos.symbols().contains(&s) {
                        let other = pos.guess_slope(&s);
                        if me.0.signum() == other.0.signum() {
                            let new = t.clone() * me.1 * other.0.abs()
                                - pos.clone() * me.0.abs() * other.1;
                            if !visited.contains(&new) {
                                todo.push(new);
                            }
                        }
                    }
                }
            }
            visited.push(t);
            if visited.len() > 10 {
                break
            }
        }
        false
    }
}

impl fmt::Debug for SymbolScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let locked = self.0.lock().unwrap();
        write!(f, "{}", locked.table.into_iter().map(|(_, s)| s).join(" "))
    }
}

#[derive(Debug, PartialEq, Clone, Hash)]
#[allow(clippy::upper_case_acronyms)]
pub enum InequalitySign {
    LT,
    GT,
    LTE,
    GTE,
}

#[derive(Debug, PartialEq, Clone, Hash)]
pub struct Inequality {
    pub left: TDim,
    pub sign: InequalitySign,
    pub right: TDim,
}

impl Inequality {
    pub fn as_known_positive(&self) -> TDim {
        use InequalitySign::*;
        match self.sign {
            GTE => self.left.clone() - &self.right,
            GT => self.left.clone() - 1 - &self.right,
            LTE => self.right.clone() - &self.left,
            LT => self.right.clone() - 1 - &self.left,
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn as_known_positive_gte() {
        let s = SymbolScope::default();
        assert_eq!(
            s.parse_inequality("S>=0").unwrap().as_known_positive(),
            s.parse_tdim("S").unwrap()
        );
    }

    #[test]
    fn as_known_positive_gt() {
        let s = SymbolScope::default();
        assert_eq!(
            s.parse_inequality("S>0").unwrap().as_known_positive(),
            s.parse_tdim("S-1").unwrap()
        );
    }

    #[test]
    fn as_known_positive_lte() {
        let s = SymbolScope::default();
        assert_eq!(
            s.parse_inequality("S<=0").unwrap().as_known_positive(),
            s.parse_tdim("-S").unwrap()
        );
    }

    #[test]
    fn as_known_positive_lt() {
        let s = SymbolScope::default();
        assert_eq!(
            s.parse_inequality("S<0").unwrap().as_known_positive(),
            s.parse_tdim("-S - 1").unwrap()
        );
    }

    #[test]
    fn prove_positive_0() {
        let s = SymbolScope::default();
        assert!(s.prove_positive(&s.parse_tdim("0").unwrap()));
    }

    #[test]
    fn prove_positive_1() {
        let s = SymbolScope::default();
        assert!(s.prove_positive(&s.parse_tdim("1").unwrap()));
    }

    #[test]
    fn prove_positive_neg1() {
        let s = SymbolScope::default();
        assert!(!s.prove_positive(&s.parse_tdim("-1").unwrap()));
    }

    #[test]
    fn prove_positive_add_0() {
        let s = SymbolScope::default();
        assert!(!s.prove_positive(&s.parse_tdim("s+1").unwrap()));
    }

    #[test]
    fn prove_positive_with_axiom() {
        let s = SymbolScope::default();
        s.add_inequality(s.parse_inequality("s>=0").unwrap());
        assert!(s.prove_positive(&s.parse_tdim("s").unwrap()));
    }

    #[test]
    fn prove_positive_with_axiom_2() {
        let s = SymbolScope::default();
        s.add_inequality(s.parse_inequality("s>=0").unwrap());
        s.add_inequality(s.parse_inequality("p>=0").unwrap());
        s.add_inequality(s.parse_inequality("p+s<4096").unwrap());
        assert!(s.prove_positive(&s.parse_tdim("4096-p").unwrap()));
    }
}
