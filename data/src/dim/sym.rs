use itertools::Itertools;
use parking_lot::ReentrantMutex;
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::fmt::{self, Display};
use std::sync::{Arc, Weak};
use string_interner::DefaultStringInterner;
use string_interner::Symbol as _;

use crate::TractResult;

use super::parse::parse_assertion;
use super::{parse_tdim, Assertion, TDim};

#[derive(Clone, Default)]
pub struct SymbolScope(pub Arc<ReentrantMutex<RefCell<SymbolScopeData>>>);

impl PartialEq for SymbolScope {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for SymbolScope {}

#[derive(Default)]
pub struct SymbolScopeData {
    table: DefaultStringInterner,
    assertions: Vec<Assertion>,
    scenarios: BTreeMap<String, Vec<Assertion>>,
}

impl SymbolScope {
    pub fn get(&self, name: &str) -> Option<Symbol> {
        let locked = self.0.lock();
        let locked = locked.borrow();
        locked.table.get(name).map(|sym| Symbol(Arc::downgrade(&self.0), sym))
    }

    pub fn sym(&self, name: &str) -> Symbol {
        let locked = self.0.lock();
        let mut locked = locked.borrow_mut();
        let sym = locked.table.get_or_intern(name);
        Symbol(Arc::downgrade(&self.0), sym)
    }

    pub fn new_with_prefix(&self, prefix: &str) -> Symbol {
        let locked = self.0.lock();
        let mut locked = locked.borrow_mut();
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
        Symbol(Arc::downgrade(&self.0), sym)
    }

    pub fn parse_tdim(&self, input: impl AsRef<str>) -> TractResult<TDim> {
        parse_tdim(self, input.as_ref())
    }

    pub fn add_assertion(&self, assert: impl Into<String>) -> TractResult<()> {
        let assert = assert.into();
        let assert = parse_assertion(self, &assert)?;
        let locked = self.0.lock();
        let mut locked = locked.borrow_mut();
        locked.assertions.push(assert);
        Ok(())
    }

    pub fn with_assertion(self, assert: impl Into<String>) -> TractResult<Self> {
        self.add_assertion(assert)?;
        Ok(self)
    }

    pub fn all_assertions(&self) -> Vec<Assertion> {
        let locked = self.0.lock();
        let locked = locked.borrow();
        locked.assertions.clone()
    }

    pub fn all_scenarios(&self) -> impl IntoIterator<Item=(String, Vec<Assertion>)> {
        let locked = self.0.lock();
        let locked = locked.borrow();
        locked.scenarios.clone()
    }

    pub fn add_scenario(&self, scenario: impl Into<String>) -> TractResult<()> {
        let locked = self.0.lock();
        let mut locked = locked.borrow_mut();
        locked.scenarios.insert(scenario.into(), vec![]);
        Ok(())
    }

    pub fn add_scenario_assertion(
        &self,
        scenario: impl Into<String>,
        assertion: impl Into<String>,
    ) -> TractResult<()> {
        let assert = parse_assertion(self, &assertion.into())?;
        let s = scenario.into();
        let locked = self.0.lock();
        let mut locked = locked.borrow_mut();
        locked.scenarios.entry(s).or_default().push(assert);
        Ok(())
    }

    pub fn with_scenario_assertion(
        self,
        scenario: impl Into<String>,
        assertion: impl Into<String>,
    ) -> TractResult<Self> {
        self.add_scenario_assertion(scenario, assertion)?;
        Ok(self)
    }

    pub fn with_scenario(self, scenario: impl Into<String>) -> TractResult<Self> {
        self.add_scenario(scenario)?;
        Ok(self)
    }

    pub fn all_symbols(&self) -> Vec<Symbol> {
        self.0
            .lock()
            .borrow()
            .table
            .into_iter()
            .map(|is| Symbol(Arc::downgrade(&self.0), is.0))
            .collect()
    }
}

impl SymbolScopeData {
    pub fn all_assertions(&self) -> &[Assertion] {
        &self.assertions
    }

    pub fn assertions(&self, scenario: Option<&str>) -> impl Iterator<Item = &'_ Assertion> {
        self.assertions.iter().chain(if let Some(s) = scenario {
            self.scenarios[s].iter()
        } else {
            [].iter()
        })
    }

    pub fn scenarios(&self) -> impl Iterator<Item = &'_ str> {
        self.scenarios.keys().map(|s| s.as_ref())
    }

    pub fn scenario(&self, s: &str) -> impl Iterator<Item = &'_ Assertion> {
        self.scenarios[s].iter()
    }

    pub fn resolving<R>(&self, sym: &Symbol, f: impl FnOnce(&str) -> R) -> Option<R> {
        self.table.resolve(sym.1).map(f)
    }

    #[allow(clippy::mutable_key_type)]
    pub fn prove_positive_or_zero(&self, t: &TDim) -> bool {
        if let TDim::Val(v) = t {
            return *v >= 0;
        }
        let positives = self.assertions.iter().filter_map(|i| i.as_known_positive()).collect_vec();
        let mut visited = vec![];
        let mut todo = vec![t.clone()];
        while let Some(t) = todo.pop() {
            if t.to_i64().is_ok_and(|i| i >= 0) {
                return true;
            }
            if t.inclusive_bound(self, false).is_some_and(|l| l >= 0) {
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
                break;
            }
        }
        false
    }
}

impl fmt::Debug for SymbolScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let locked = self.0.lock();
        let locked = locked.borrow();
        write!(f, "{}", locked.table.into_iter().map(|(_, s)| s).join(" "))
    }
}

#[derive(Clone)]
pub struct Symbol(Weak<ReentrantMutex<RefCell<SymbolScopeData>>>, string_interner::DefaultSymbol);

impl Eq for Symbol {}

impl PartialEq for Symbol {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}

impl Symbol {
    pub fn scope(&self) -> Option<SymbolScope> {
        self.0.upgrade().map(SymbolScope)
    }
}

impl PartialOrd for Symbol {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.1.cmp(&other.1))
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
        if let Some(scope) = self.scope() {
            let lock = scope.0.lock();
            let lock = lock.borrow();
            if let Some(s) = lock.table.resolve(self.1) {
                return write!(f, "{}", s);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn as_known_positive_gte() {
        let s = SymbolScope::default();
        assert_eq!(
            parse_assertion(&s, "S>=0").unwrap().as_known_positive(),
            Some(s.parse_tdim("S").unwrap())
        );
    }

    #[test]
    fn as_known_positive_gt() {
        let s = SymbolScope::default();
        assert_eq!(
            parse_assertion(&s, "S>0").unwrap().as_known_positive(),
            Some(s.parse_tdim("S-1").unwrap())
        );
    }

    #[test]
    fn as_known_positive_lte() {
        let s = SymbolScope::default();
        assert_eq!(
            parse_assertion(&s, "S<=0").unwrap().as_known_positive(),
            Some(s.parse_tdim("-S").unwrap())
        );
    }

    #[test]
    fn as_known_positive_lt() {
        let s = SymbolScope::default();
        assert_eq!(
            parse_assertion(&s, "S<0").unwrap().as_known_positive(),
            Some(s.parse_tdim("-S - 1").unwrap())
        );
    }

    #[test]
    fn prove_positive_0() {
        let s = SymbolScope::default();
        assert!(s.parse_tdim("0").unwrap().prove_positive_or_zero());
    }

    #[test]
    fn prove_positive_1() {
        let s = SymbolScope::default();
        assert!(s.parse_tdim("1").unwrap().prove_positive_or_zero());
    }

    #[test]
    fn prove_positive_neg1() {
        let s = SymbolScope::default();
        assert!(!s.parse_tdim("-1").unwrap().prove_positive_or_zero());
    }

    #[test]
    fn prove_positive_add_0() {
        let s = SymbolScope::default();
        assert!(!s.parse_tdim("s+1").unwrap().prove_positive_or_zero());
    }
}
