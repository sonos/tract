use crate::dim::Assertion;
use crate::internal::*;

use super::{sym::*, DimLike};
use itertools::Itertools;
use num_integer::Integer;
use num_traits::{AsPrimitive, PrimInt, Zero};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::ops::Neg;
use std::{fmt, ops};

#[derive(Debug)]
pub enum TooEarly {
    UndeterminedSymbol(TDim),
    Other(String),
}

impl std::fmt::Display for TooEarly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TooEarly::UndeterminedSymbol(s) => write!(f, "Undetermined symbol in expression: {s}"),
            TooEarly::Other(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for TooEarly {}

macro_rules! b( ($e:expr) => { Box::new($e) } );

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum TDim {
    Val(i64),
    Sym(Symbol),
    Add(Vec<TDim>),
    Mul(Vec<TDim>),
    MulInt(i64, Box<TDim>),
    Div(Box<TDim>, u64),
    Broadcast(Vec<TDim>),
    Min(Vec<TDim>),
    Max(Vec<TDim>),
}

use TDim::*;

fn tdim_lexi_order(a: &TDim, b: &TDim) -> Ordering {
    match (a, b) {
        (Sym(a), Sym(b)) => a.cmp(b),
        (Val(a), Val(b)) => a.cmp(b),
        (Add(a), Add(b))
        | (Mul(a), Mul(b))
        | (Broadcast(a), Broadcast(b))
        | (Min(a), Min(b))
        | (Max(a), Max(b)) => a.len().cmp(&b.len()).then(
            a.iter()
                .zip(b.iter())
                .fold(Ordering::Equal, |acc, (a, b)| acc.then_with(|| tdim_lexi_order(a, b))),
        ),
        (MulInt(p, d), MulInt(q, e)) => p.cmp(q).then_with(|| tdim_lexi_order(d, e)),
        (Div(d, p), Div(e, q)) => p.cmp(q).then_with(|| tdim_lexi_order(d, e)),
        (Sym(_), _) => Ordering::Less,
        (_, Sym(_)) => Ordering::Greater,
        (Val(_), _) => Ordering::Less,
        (_, Val(_)) => Ordering::Greater,
        (Add(_), _) => Ordering::Less,
        (_, Add(_)) => Ordering::Greater,
        (Mul(_), _) => Ordering::Less,
        (_, Mul(_)) => Ordering::Greater,
        (MulInt(_, _), _) => Ordering::Less,
        (_, MulInt(_, _)) => Ordering::Greater,
        (Broadcast(_), _) => Ordering::Less,
        (_, Broadcast(_)) => Ordering::Greater,
        (Min(_), _) => Ordering::Less,
        (_, Min(_)) => Ordering::Greater,
        (Max(_), _) => Ordering::Less,
        (_, Max(_)) => Ordering::Greater,
    }
}

impl fmt::Display for TDim {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match &self {
            Sym(sym) => write!(fmt, "{sym}"),
            Val(it) => write!(fmt, "{it}"),
            Add(it) => write!(fmt, "{}", it.iter().map(|x| format!("{x}")).join("+")),
            Mul(it) => write!(fmt, "{}", it.iter().map(|x| format!("({x})")).join("*")),
            Broadcast(it) => write!(fmt, "{}", it.iter().map(|x| format!("({x})")).join("#")),
            Min(it) => write!(fmt, "min({})", it.iter().map(|x| format!("{x}")).join(",")),
            Max(it) => write!(fmt, "max({})", it.iter().map(|x| format!("{x}")).join(",")),
            MulInt(a, b) => write!(fmt, "{a}*{b}"),
            Div(a, b) => write!(fmt, "({a})/{b}"),
        }
    }
}

impl TDim {
    #[inline]
    pub fn is_one(&self) -> bool {
        matches!(self, Val(1))
    }

    #[inline]
    pub fn to_i64(&self) -> TractResult<i64> {
        if let Val(v) = self {
            Ok(*v)
        } else {
            Err(TooEarly::UndeterminedSymbol(self.clone()).into())
        }
    }

    #[inline]
    pub fn as_i64(&self) -> Option<i64> {
        if let Val(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    pub fn eval_to_i64(&self, values: &SymbolValues) -> TractResult<i64> {
        match self {
            Sym(sym) => {
                let Some(v) = values.get(sym) else {
                    bail!(TooEarly::UndeterminedSymbol(self.clone()))
                };
                Ok(v)
            }
            Val(v) => Ok(*v),
            Add(terms) => {
                terms.iter().try_fold(0, |acc, it| it.eval_to_i64(values).map(|x| acc + x))
            }
            Mul(terms) => {
                terms.iter().try_fold(1, |acc, it| it.eval_to_i64(values).map(|x| acc * x))
            }
            Min(terms) => terms
                .iter()
                .try_fold(i64::MAX, |acc, it| it.eval_to_i64(values).map(|x| acc.min(x))),
            Max(terms) => terms
                .iter()
                .try_fold(i64::MIN, |acc, it| it.eval_to_i64(values).map(|x| acc.max(x))),
            Broadcast(terms) => terms.iter().try_fold(1i64, |acc, it| {
                it.eval_to_i64(values)
                    .and_then(|x| ((acc as usize).broadcast(x as usize)).map(|x| x as i64))
            }),
            Div(a, q) => Ok(a.eval_to_i64(values)? / *q as i64),
            MulInt(p, a) => Ok(a.eval_to_i64(values)? * *p),
        }
    }

    pub fn eval(&self, values: &SymbolValues) -> TDim {
        match self {
            Sym(sym) => values.get(sym).map(Val).unwrap_or_else(|| Sym(sym.clone())),
            Val(v) => Val(*v),
            Add(terms) => terms.iter().fold(Val(0), |acc, it| -> TDim { acc + it.eval(values) }),
            Mul(terms) => terms.iter().fold(Val(1), |acc, it| -> TDim { acc * it.eval(values) }),
            Min(terms) => {
                terms.iter().fold(Val(i64::MAX), |acc, it| -> TDim { acc.mini(it.eval(values)) })
            }
            Max(terms) => {
                terms.iter().fold(Val(i64::MIN), |acc, it| -> TDim { acc.maxi(it.eval(values)) })
            }
            Broadcast(terms) => terms.iter().fold(Val(1), |acc, it| -> TDim {
                acc.broadcast(it.eval(values)).unwrap_or_else(|_| self.clone())
            }),
            Div(a, q) => a.eval(values) / *q as i64,
            MulInt(p, a) => a.eval(values) * *p,
        }
    }

    pub fn eval_with_scenario(&self, scenario: &str) -> TDim {
        if let Val(v) = self {
            return Val(*v);
        }
        let scope = self.find_scope().unwrap();
        let scope = scope.0;
        let locked = scope.lock();
        let scope = locked.borrow();
        self.clone().simplify_rec(&scope, Some(scenario))
    }

    pub fn substitute(&self, from: &Symbol, to: &Self) -> TractResult<Self> {
        match self {
            Sym(sym) => Ok(if sym == from { to.clone() } else { self.clone() }),
            Val(v) => Ok(Val(*v)),
            Add(terms) => terms.iter().try_fold(Val(0), |acc, it| -> TractResult<TDim> {
                Ok(acc + it.substitute(from, to)?)
            }),
            Mul(terms) => terms.iter().try_fold(Val(1), |acc, it| -> TractResult<TDim> {
                Ok(acc * it.substitute(from, to)?)
            }),
            Broadcast(terms) => terms.iter().try_fold(Val(1), |acc, it| -> TractResult<TDim> {
                acc.broadcast(it.substitute(from, to)?)
            }),
            Min(terms) => terms.iter().try_fold(Val(i64::MAX), |acc, it| -> TractResult<TDim> {
                Ok(acc.mini(it.substitute(from, to)?))
            }),
            Max(terms) => terms.iter().try_fold(Val(i64::MIN), |acc, it| -> TractResult<TDim> {
                Ok(acc.maxi(it.substitute(from, to)?))
            }),
            Div(a, q) => Ok(a.substitute(from, to)? / *q as i64),
            MulInt(p, a) => Ok(a.substitute(from, to)? * *p),
        }
    }

    pub fn reduce(self) -> TDim {
        self.simplify()
            .wiggle()
            .into_iter()
            .sorted_by(tdim_lexi_order)
            .unique()
            .map(|e| e.simplify())
            .min_by_key(|e| e.cost())
            .unwrap()
    }

    fn cost(&self) -> usize {
        use self::TDim::*;
        match self {
            Sym(_) | Val(_) => 1,
            Add(terms) => 2 * terms.iter().map(TDim::cost).sum::<usize>(),
            Mul(terms) => 3 * terms.iter().map(TDim::cost).sum::<usize>(),
            Broadcast(terms) => 4 * terms.iter().map(TDim::cost).sum::<usize>(),
            Min(terms) | Max(terms) => 5 * terms.iter().map(TDim::cost).sum::<usize>(),
            Div(a, _) => 3 * a.cost(),
            MulInt(_, a) => 2 * a.cost(),
        }
    }

    fn wiggle(&self) -> Vec<TDim> {
        use self::TDim::*;
        match self {
            Sym(_) | Val(_) | Mul(_) | Broadcast(_) | Min(_) | Max(_) => vec![self.clone()],
            Add(terms) => {
                let mut forms = vec![];
                let sub_exprs = terms.iter().map(|e| e.wiggle()).multi_cartesian_product();

                fn first_div_term(terms: &[TDim]) -> Option<(usize, &TDim, u64)> {
                    terms.iter().enumerate().find_map(|(index, t)| match t {
                        Div(numerator, quotient) => Some((index, &**numerator, *quotient)),
                        _ => None,
                    })
                }

                fn generate_new_numerator(
                    div_index: usize,
                    numerator: &TDim,
                    quotient: u64,
                    expr: &[TDim],
                ) -> Vec<TDim> {
                    expr.iter()
                        .enumerate()
                        .map(|(index, term)| {
                            if index == div_index {
                                numerator.clone()
                            } else {
                                MulInt(quotient as i64, Box::new(term.clone()))
                            }
                        })
                        .collect()
                }

                for expr in sub_exprs {
                    if let Some((div_index, numerator, quotient)) = first_div_term(&expr) {
                        let new_numerator =
                            generate_new_numerator(div_index, numerator, quotient, &expr);
                        forms.push(Div(Box::new(Add(new_numerator)), quotient))
                    }

                    forms.push(Add(expr));
                }
                forms
            }
            MulInt(p, a) => a.wiggle().into_iter().map(|a| MulInt(*p, b!(a))).collect(),
            Div(a, q) => {
                let mut forms = vec![];
                for num in a.wiggle() {
                    if let Add(terms) = &num {
                        let (integer, non_integer): (Vec<_>, Vec<_>) =
                            terms.iter().cloned().partition(|a| a.gcd() % q == 0);
                        let mut new_terms = integer.iter().map(|i| i.div(*q)).collect::<Vec<_>>();
                        if non_integer.len() > 0 {
                            new_terms.push(Div(b!(Add(non_integer)), *q));
                        }
                        forms.push(Add(new_terms))
                    }
                    forms.push(Div(b!(num), *q))
                }
                forms
            }
        }
    }

    fn find_any_sym(tdim: &TDim) -> Option<&Symbol> {
        match tdim {
            Val(_) => None,
            Sym(s) => Some(s),
            Add(terms) | Mul(terms) | Min(terms) | Max(terms) | Broadcast(terms) => {
                terms.iter().find_map(Self::find_any_sym)
            }
            MulInt(_, t) | Div(t, _) => Self::find_any_sym(t),
        }
    }

    pub fn find_scope(&self) -> Option<SymbolScope> {
        Self::find_any_sym(self).and_then(|s| s.scope().clone())
    }

    pub fn simplify(self) -> TDim {
        use self::TDim::*;
        if let Ok(v) = self.eval_to_i64(&SymbolValues::default()) {
            return Val(v);
        }
        let Some(scope) = self.find_scope() else {
            return self;
        };
        let scope = scope.0;
        let locked = scope.lock();
        let scope = locked.borrow();
        let it = self.simplify_rec(&scope, None);
        let mut current: Option<TDim> = None;
        for scenario in scope.scenarios() {
            let v = it.clone().simplify_rec(&scope, Some(scenario));
            if current.is_some_and(|c| c != v) {
                return it;
            } else {
                current = Some(v);
            }
        }
        current.unwrap_or(it)
    }

    fn simplify_rec(self, scope: &SymbolScopeData, scenario: Option<&str>) -> TDim {
        match self {
            Add(mut terms) => {
                #[allow(clippy::mutable_key_type)]
                let mut simplified_terms: HashMap<TDim, i64> = HashMap::new();
                // factorize common sub-expr
                while let Some(term) = terms.pop() {
                    let simplified = term.simplify_rec(scope, scenario);
                    match simplified {
                        Val(0) => {} // ignore
                        Add(members) => {
                            terms.extend(members);
                            continue;
                        }
                        Val(value) => *simplified_terms.entry(Val(1)).or_insert(0) += value,
                        MulInt(value, factor) => {
                            *simplified_terms.entry((*factor).clone()).or_insert(0) += value;
                        }
                        n => *simplified_terms.entry(n).or_insert(0) += 1,
                    };
                }

                pub fn evaluate_count(term: TDim, count: i64) -> Option<TDim> {
                    match count {
                        0 => None,
                        _ if term == TDim::Val(1) => Some(TDim::Val(count)),
                        1 => Some(term),
                        _ => Some(TDim::MulInt(count, Box::new(term))),
                    }
                }

                let mut members: Vec<TDim> = simplified_terms
                    .into_iter()
                    .filter_map(|(term, count)| evaluate_count(term, count))
                    .collect();
                members.sort_by(tdim_lexi_order);

                match members.len() {
                    0 => TDim::Val(0),
                    1 => members.into_iter().next().unwrap(),
                    _ => TDim::Add(members),
                }
            }
            Mul(terms) => {
                let mut gcd = Mul(terms.clone()).gcd() as i64;
                if gcd == 0 {
                    return Val(0);
                }
                let mut terms = if gcd != 1 {
                    terms
                        .into_iter()
                        .map(|t| {
                            let gcd = t.gcd();
                            (t / gcd).simplify_rec(scope, scenario)
                        })
                        .collect()
                } else {
                    terms
                };
                if terms.iter().filter(|t| t == &&Val(-1)).count() % 2 == 1 {
                    gcd = -gcd;
                }
                terms.retain(|t| !t.is_one() && t != &Val(-1));
                terms.sort_by(tdim_lexi_order);
                match (gcd, terms.len()) {
                    (_, 0) => Val(gcd), // Case #1: If 0 variables, return product
                    (0, _) => Val(0),   // Case #2: Result is 0 if coef is 0 (actually
                    // unreachable as we check at the beginning)
                    (1, 1) => terms.remove(0), // Case #3: Product is 1, so return the only term
                    (1, _) => Mul(terms), // Case #4: Product is 1, so return the non-integer terms
                    (_, 1) => MulInt(gcd, Box::new(terms.remove(0))), // Case #5: Single variable, convert to 1 MulInt
                    _ => MulInt(gcd, Box::new(Mul(terms))), // Case #6: Multiple variables, convert to MulInt
                }
            }
            MulInt(coef, expr) => {
                match *expr {
                    MulInt(c2, inner) => {
                        return MulInt(coef * c2, inner).simplify_rec(scope, scenario)
                    }
                    Val(v) => return Val(coef * v),
                    _ => {}
                }

                let simplified = expr.simplify_rec(scope, scenario);
                match (coef, simplified) {
                    (0, _) => Val(0), // Case #1: If coef is 0, return 0
                    (1, s) => s,      // Case #2: If coef is 1, return the simplified expression
                    (_, Add(terms)) => Add(terms
                        .into_iter()
                        .map(|term| MulInt(coef, Box::new(term)).simplify_rec(scope, scenario))
                        .collect()), // Case #3: If expression is an addition, distribute the coef
                    (c, Val(v)) => Val(c * v), // Case #4: If expression is a value, combine coefs
                    (c, MulInt(v, inner)) => MulInt(c * v, inner), // Case #5: If expression is a MulInt, combine coefs
                    (_, s) => MulInt(coef, Box::new(s)), // Case #6: Otherwise, return the original
                }
            }
            Div(a, q) => {
                if q == 1 {
                    return a.simplify_rec(scope, scenario);
                } else if let Div(a, q2) = *a {
                    return Div(a, q * q2).simplify_rec(scope, scenario);
                }
                let a = a.simplify_rec(scope, scenario);
                if let Val(a) = a {
                    Val(a / q as i64)
                } else if let MulInt(-1, a) = a {
                    MulInt(-1, b!(Div(a, q)))
                } else if let Add(mut terms) = a {
                    if terms.iter().any(|t| {
                        if let MulInt(-1, s) = t {
                            matches!(&**s, Sym(_))
                        } else {
                            false
                        }
                    }) {
                        MulInt(
                            -1,
                            b!(Div(
                                b!(Add(terms.into_iter().map(|t| MulInt(-1, b!(t))).collect())
                                    .simplify_rec(scope, scenario)),
                                q
                            )),
                        )
                    } else if let Some(v) =
                        terms.iter().find_map(|t| if let Val(v) = t { Some(*v) } else { None })
                    {
                        let offset = if v >= q as i64 {
                            Some(v / q as i64)
                        } else if v < 0 {
                            Some(-Integer::div_ceil(&-v, &(q as i64)))
                        } else {
                            None
                        };
                        if let Some(val) = offset {
                            terms.push(Val(-val * q as i64));
                            Add(vec![
                                Val(val),
                                Div(b!(Add(terms).simplify_rec(scope, scenario)), q),
                            ])
                        } else {
                            Div(b!(Add(terms)), q)
                        }
                    } else {
                        Div(b!(Add(terms)), q)
                    }
                } else if let MulInt(p, a) = a {
                    if p == q as i64 {
                        a.simplify()
                    } else {
                        let gcd = p.abs().gcd(&(q as i64));
                        if gcd == p {
                            Div(a, q / gcd as u64)
                        } else if gcd == q as i64 {
                            MulInt(p / gcd, a)
                        } else if gcd > 1 {
                            Div(b!(MulInt(p / gcd, a)), q / gcd as u64)
                                .simplify_rec(scope, scenario)
                        } else {
                            Div(b!(MulInt(p, a)), q)
                        }
                    }
                } else {
                    Div(b!(a), q)
                }
            }
            Broadcast(terms) => {
                let mut terms: Vec<TDim> = terms
                    .iter()
                    .map(|s| s.clone().simplify_rec(scope, scenario))
                    .flat_map(|t| if let Broadcast(t) = t { t } else { vec![t] })
                    .filter(|t| !t.is_one())
                    .sorted_by(tdim_lexi_order)
                    .dedup()
                    .collect_vec();
                if terms.len() == 0 {
                    Val(1)
                } else if terms.len() == 1 {
                    terms.remove(0)
                } else {
                    Broadcast(terms)
                }
            }

            Min(terms) => {
                let mut flatten: Vec<TDim> = terms
                    .into_iter()
                    .map(|t| t.simplify_rec(scope, scenario))
                    .flat_map(|t| if let Min(t) = t { t } else { vec![t] })
                    .sorted_by(tdim_lexi_order)
                    .dedup()
                    .collect();
                #[allow(clippy::mutable_key_type)]
                let mut redundant = HashSet::<TDim>::default();
                for pair in flatten.iter().permutations(2) {
                    let (a, b) = (pair[0], pair[1]);
                    if redundant.contains(a) || redundant.contains(b) {
                        continue;
                    }
                    let diff = a.clone() - b;
                    if diff.as_i64().is_some_and(|i| i >= 0) || scope.prove_positive_or_zero(&diff)
                    {
                        redundant.insert(a.clone());
                    }
                }
                flatten.retain(|t| !redundant.contains(t));
                if flatten.len() == 0 {
                    i64::MAX.to_dim()
                } else if flatten.len() == 1 {
                    flatten.into_iter().next().unwrap()
                } else {
                    Min(flatten)
                }
            }
            Max(terms) => {
                let mut flatten: Vec<TDim> = terms
                    .into_iter()
                    .map(|t| t.simplify_rec(scope, scenario))
                    .flat_map(|t| if let Max(t) = t { t } else { vec![t] })
                    .sorted_by(tdim_lexi_order)
                    .dedup()
                    .collect();
                #[allow(clippy::mutable_key_type)]
                let mut redundant = HashSet::<TDim>::default();
                for pair in flatten.iter().permutations(2) {
                    let (a, b) = (pair[0], pair[1]);
                    if redundant.contains(a) || redundant.contains(b) {
                        continue;
                    }
                    let diff = a.clone() - b;
                    if diff.as_i64().is_some_and(|i| i >= 0) || scope.prove_positive_or_zero(&diff)
                    {
                        redundant.insert(b.clone());
                    }
                }
                flatten.retain(|t| !redundant.contains(t));
                if flatten.len() == 0 {
                    i64::MIN.to_dim()
                } else if flatten.len() == 1 {
                    flatten.into_iter().next().unwrap()
                } else {
                    Max(flatten)
                }
            }
            Sym(s) => scope
                .assertions(scenario)
                .find_map(|a| match a {
                    Assertion::Eq(Sym(sym), v) if sym == &s => Some(v.clone()),
                    _ => None,
                })
                .unwrap_or(Sym(s)),
            Val(_) => self,
        }
    }

    pub(super) fn inclusive_bound(&self, scope: &SymbolScopeData, upper: bool) -> Option<i64> {
        use self::TDim::*;
        match self {
            Val(n) => Some(*n),
            Sym(_) => {
                if upper {
                    scope
                        .all_assertions()
                        .iter()
                        .filter_map(|assert| match &assert {
                            Assertion::LT(left, right)
                                if left == self && right.as_i64().is_some() =>
                            {
                                Some(right.as_i64().unwrap() - 1)
                            }
                            Assertion::LTE(left, right)
                                if left == self && right.as_i64().is_some() =>
                            {
                                Some(right.as_i64().unwrap())
                            }
                            _ => None,
                        })
                        .min()
                } else {
                    scope
                        .all_assertions()
                        .iter()
                        .filter_map(|assert| match &assert {
                            Assertion::GT(left, right)
                                if left == self && right.as_i64().is_some() =>
                            {
                                Some(right.as_i64().unwrap() + 1)
                            }
                            Assertion::GTE(left, right)
                                if left == self && right.as_i64().is_some() =>
                            {
                                Some(right.as_i64().unwrap())
                            }
                            _ => None,
                        })
                        .max()
                }
            }
            Add(terms) => {
                let mut bound = 0;
                for t in terms {
                    if let Some(b) = t.inclusive_bound(scope, upper) {
                        bound += b;
                    } else {
                        return None;
                    }
                }
                Some(bound)
            }
            MulInt(p, a) => match p.cmp(&0) {
                Ordering::Equal => Some(0),
                Ordering::Greater => a.inclusive_bound(scope, upper).map(|x| x * p),
                Ordering::Less => a.inclusive_bound(scope, !upper).map(|x| x * p),
            },
            Mul(_) => None,
            Min(terms) if !upper => {
                terms.iter().filter_map(|t| t.inclusive_bound(scope, false)).min()
            }
            Max(terms) if upper => {
                terms.iter().filter_map(|t| t.inclusive_bound(scope, true)).max()
            }
            Div(a, q) => a.inclusive_bound(scope, upper).map(|x| x / (*q as i64)),
            Broadcast(terms) => {
                if upper {
                    Max(terms.clone()).inclusive_bound(scope, true)
                } else {
                    Min(terms.clone()).inclusive_bound(scope, false)
                }
            }
            _ => None,
        }
    }

    pub fn low_inclusive_bound(&self) -> Option<i64> {
        if let TDim::Val(v) = self {
            return Some(*v);
        }
        let scope = self.find_scope()?;
        let data = scope.0.lock();
        let data = data.borrow();
        self.inclusive_bound(&data, false)
    }

    pub fn high_inclusive_bound(&self) -> Option<i64> {
        if let TDim::Val(v) = self {
            return Some(*v);
        }
        let scope = self.find_scope()?;
        let data = scope.0.lock();
        let data = data.borrow();
        self.inclusive_bound(&data, true)
    }

    pub fn prove_positive_or_zero(&self) -> bool {
        if let TDim::Val(v) = self {
            return *v >= 0;
        }
        let Some(scope) = self.find_scope() else { return false };
        let data = scope.0.lock();
        let data = data.borrow();
        data.prove_positive_or_zero(self)
    }

    pub fn prove_strict_positive(&self) -> bool {
        if let TDim::Val(v) = self {
            return *v > 0;
        }
        (self.clone() - 1).prove_positive_or_zero()
    }

    pub fn prove_negative_or_zero(&self) -> bool {
        if let TDim::Val(v) = self {
            return *v <= 0;
        }
        self.clone().neg().prove_positive_or_zero()
    }

    pub fn prove_strict_negative(&self) -> bool {
        if let TDim::Val(v) = self {
            return *v < 0;
        }
        self.clone().neg().prove_strict_positive()
    }

    pub fn gcd(&self) -> u64 {
        use self::TDim::*;
        match self {
            Val(v) => v.unsigned_abs(),
            Sym(_) => 1,
            Add(terms) => {
                let (head, tail) = terms.split_first().unwrap();
                tail.iter().fold(head.gcd(), |a, b| a.gcd(&b.gcd()))
            }
            MulInt(p, a) => a.gcd() * p.unsigned_abs(),
            Mul(terms) => terms.iter().map(|t| t.gcd()).product(),
            Min(terms) => terms.iter().map(|t| t.gcd()).reduce(|a, b| a.gcd(&b)).unwrap(),
            Max(terms) => terms.iter().map(|t| t.gcd()).reduce(|a, b| a.gcd(&b)).unwrap(),
            Div(a, q) => {
                if a.gcd() % *q == 0 {
                    a.gcd() / *q
                } else {
                    1
                }
            }
            Broadcast(terms) => terms.iter().map(|t| t.gcd()).reduce(|a, b| a.gcd(&b)).unwrap_or(1),
        }
    }

    fn div(&self, d: u64) -> TDim {
        use self::TDim::*;
        if d == 1 {
            return self.clone();
        }
        match self {
            Val(v) => Val(v / d as i64),
            Sym(_) => panic!(),
            Add(terms) => Add(terms.iter().map(|t| t.div(d)).collect()),
            Min(terms) => Min(terms.iter().map(|t| t.div(d)).collect()),
            Max(terms) => Max(terms.iter().map(|t| t.div(d)).collect()),
            Broadcast(terms) => Broadcast(terms.iter().map(|t| t.div(d)).collect()),
            Mul(_) => Div(Box::new(self.clone()), d),
            MulInt(p, a) => {
                if *p == d as i64 {
                    (**a).clone()
                } else {
                    let gcd = p.unsigned_abs().gcd(&d);
                    MulInt(p / gcd as i64, b!(a.div(d / gcd)))
                }
            }
            Div(a, q) => Div(a.clone(), q * d),
        }
    }

    pub fn div_ceil(self, rhs: u64) -> TDim {
        TDim::Div(Box::new(Add(vec![self, Val(rhs as i64 - 1)])), rhs).reduce()
    }

    pub(super) fn guess_slope(&self, sym: &Symbol) -> (i64, u64) {
        fn slope_rec(d: &TDim, sym: &Symbol) -> (i64, i64) {
            match d {
                Val(_) => (0, 1),
                Sym(s) => ((sym == s) as i64, 1),
                Add(terms) => terms
                    .iter()
                    .map(|d| slope_rec(d, sym))
                    .fold((0, 1), |a, b| ((a.0 * b.1 + a.1 * b.0), (b.1 * a.1))),
                Mul(terms) => terms
                    .iter()
                    .map(|d| slope_rec(d, sym))
                    .fold((1, 1), |a, b| ((a.0 * b.0), (b.1 * a.1))),
                MulInt(p, a) => {
                    let (n, d) = slope_rec(a, sym);
                    (p * n, d)
                }
                Div(a, q) => {
                    let (n, d) = slope_rec(a, sym);
                    (n, d * *q as i64)
                }
                Broadcast(terms) => slope_rec(&terms[0], sym),
                Min(terms) => slope_rec(&terms[0], sym),
                Max(terms) => slope_rec(&terms[0], sym),
            }
        }
        let (p, q) = slope_rec(self, sym);
        reduce_ratio(p, q)
    }

    #[allow(clippy::mutable_key_type)]
    pub fn symbols(&self) -> std::collections::HashSet<Symbol> {
        match self {
            Val(_) => maplit::hashset!(),
            Sym(s) => maplit::hashset!(s.clone()),
            Add(terms) | Mul(terms) | Broadcast(terms) | Min(terms) | Max(terms) => {
                terms.iter().fold(maplit::hashset!(), |mut set, v| {
                    set.extend(v.symbols());
                    set
                })
            }
            MulInt(_, a) => a.symbols(),
            Div(a, _) => a.symbols(),
        }
    }

    pub fn compatible_with(&self, other: &TDim) -> bool {
        if let Ok(x) = (self.clone() - other).to_i64() {
            return x == 0;
        }
        true // maybe ? :)
    }
}

pub(super) fn reduce_ratio(mut p: i64, mut q: i64) -> (i64, u64) {
    let gcd = p.abs().gcd(&q.abs());
    if gcd > 1 {
        p /= gcd;
        q /= gcd;
    }
    if q < 0 {
        (-p, (-q) as u64)
    } else {
        (p, q as u64)
    }
}

impl Zero for TDim {
    fn zero() -> Self {
        Val(0)
    }
    fn is_zero(&self) -> bool {
        matches!(self, Val(0))
    }
}

impl Default for TDim {
    fn default() -> TDim {
        Val(0)
    }
}

impl num_traits::Bounded for TDim {
    fn min_value() -> Self {
        TDim::Val(i64::MIN)
    }

    fn max_value() -> Self {
        TDim::Val(i64::MAX)
    }
}

impl num_traits::One for TDim {
    fn one() -> Self {
        TDim::Val(1)
    }
}

impl ::std::iter::Sum for TDim {
    fn sum<I: Iterator<Item = TDim>>(iter: I) -> TDim {
        iter.fold(0.into(), |a, b| a + b)
    }
}

impl<'a> ::std::iter::Sum<&'a TDim> for TDim {
    fn sum<I: Iterator<Item = &'a TDim>>(iter: I) -> TDim {
        iter.fold(0.into(), |a, b| a + b)
    }
}

impl std::iter::Product for TDim {
    fn product<I: Iterator<Item = TDim>>(iter: I) -> Self {
        iter.fold(TDim::Val(1), |a, b| a * b)
    }
}

impl<'a> ::std::iter::Product<&'a TDim> for TDim {
    fn product<I: Iterator<Item = &'a TDim>>(iter: I) -> TDim {
        iter.fold(1.into(), |a, b| a * b)
    }
}

macro_rules! from_i {
    ($i: ty) => {
        impl From<$i> for TDim {
            fn from(v: $i) -> TDim {
                TDim::Val(v as _)
            }
        }
        impl<'a> From<&'a $i> for TDim {
            fn from(v: &'a $i) -> TDim {
                TDim::Val(*v as _)
            }
        }
    };
}

from_i!(i32);
from_i!(i64);
from_i!(u64);
from_i!(isize);
from_i!(usize);

impl From<Symbol> for TDim {
    fn from(it: Symbol) -> Self {
        TDim::Sym(it)
    }
}

impl<'a> From<&'a Symbol> for TDim {
    fn from(it: &'a Symbol) -> Self {
        TDim::Sym(it.clone())
    }
}

impl ops::Neg for TDim {
    type Output = Self;
    fn neg(self) -> Self {
        if let Val(v) = self {
            Val(-v)
        } else {
            TDim::MulInt(-1, Box::new(self)).reduce()
        }
    }
}

impl<'a> ops::AddAssign<&'a TDim> for TDim {
    fn add_assign(&mut self, rhs: &'a TDim) {
        if rhs.is_zero() {
        } else if self.is_zero() {
            *self = rhs.clone();
        } else if let (Val(s), Val(o)) = (&mut *self, &rhs) {
            *s += o;
        } else {
            *self = TDim::Add(vec![std::mem::take(self), rhs.clone()]).reduce()
        }
    }
}

impl<I> ops::AddAssign<I> for TDim
where
    I: Into<TDim>,
{
    fn add_assign(&mut self, rhs: I) {
        let rhs = rhs.into();
        if rhs.is_zero() {
        } else if self.is_zero() {
            *self = rhs;
        } else if let (Val(s), Val(o)) = (&mut *self, &rhs) {
            *s += o;
        } else {
            *self = TDim::Add(vec![std::mem::take(self), rhs]).reduce()
        }
    }
}

impl<I> ops::Add<I> for TDim
where
    I: Into<TDim>,
{
    type Output = Self;
    fn add(mut self, rhs: I) -> Self {
        self += rhs;
        self
    }
}

impl<'a> ops::Add<&'a TDim> for TDim {
    type Output = Self;
    fn add(mut self, rhs: &'a TDim) -> Self {
        self += rhs;
        self
    }
}

#[allow(clippy::suspicious_op_assign_impl)]
impl<'a> ops::SubAssign<&'a TDim> for TDim {
    fn sub_assign(&mut self, rhs: &'a TDim) {
        if rhs.is_zero() {
        } else if self.is_zero() {
            *self = rhs.clone().neg();
        } else if let (Val(s), Val(o)) = (&mut *self, &rhs) {
            *s -= o;
        } else {
            *self = TDim::Add(vec![std::mem::take(self), rhs.clone().neg()]).reduce()
        }
    }
}

impl<I> ops::SubAssign<I> for TDim
where
    I: Into<TDim>,
{
    fn sub_assign(&mut self, rhs: I) {
        let rhs = rhs.into();
        if rhs.is_zero() {
        } else if self.is_zero() {
            *self = rhs.neg();
        } else if let (Val(s), Val(o)) = (&mut *self, &rhs) {
            *s -= o;
        } else {
            *self = TDim::Add(vec![std::mem::take(self), rhs.neg()]).reduce()
        }
    }
}

impl<I> ops::Sub<I> for TDim
where
    I: Into<TDim>,
{
    type Output = Self;
    fn sub(mut self, rhs: I) -> Self {
        self -= rhs;
        self
    }
}

impl<'a> ops::Sub<&'a TDim> for TDim {
    type Output = Self;
    fn sub(mut self, rhs: &'a TDim) -> Self {
        self -= rhs;
        self
    }
}

impl<I: Into<TDim>> ops::MulAssign<I> for TDim {
    fn mul_assign(&mut self, rhs: I) {
        let rhs = rhs.into();
        if self.is_one() {
            *self = rhs
        } else if rhs.is_one() {
        } else {
            *self = TDim::Mul(vec![rhs, std::mem::take(self)]).reduce()
        }
    }
}

impl<'a> ops::MulAssign<&'a TDim> for TDim {
    fn mul_assign(&mut self, rhs: &'a TDim) {
        if self.is_one() {
            *self = rhs.clone()
        } else if rhs.is_one() {
        } else {
            *self = TDim::Mul(vec![std::mem::take(self), rhs.clone()]).reduce()
        }
    }
}

impl<I: Into<TDim>> ops::Mul<I> for TDim {
    type Output = Self;
    fn mul(mut self, rhs: I) -> Self {
        self *= rhs.into();
        self
    }
}

impl<'a> ops::Mul<&'a TDim> for TDim {
    type Output = Self;
    fn mul(mut self, rhs: &'a TDim) -> Self {
        self *= rhs;
        self
    }
}

impl<I: AsPrimitive<u64> + PrimInt> ops::DivAssign<I> for TDim {
    fn div_assign(&mut self, rhs: I) {
        *self = TDim::Div(Box::new(std::mem::take(self)), rhs.as_()).reduce()
    }
}

impl<I: AsPrimitive<u64> + PrimInt> ops::Div<I> for TDim {
    type Output = Self;
    fn div(mut self, rhs: I) -> Self {
        self /= rhs.as_();
        self
    }
}

impl<I: AsPrimitive<u64> + PrimInt> ops::RemAssign<I> for TDim {
    fn rem_assign(&mut self, rhs: I) {
        *self += -(self.clone() / rhs.as_() * rhs.as_());
    }
}

impl<I: AsPrimitive<u64> + PrimInt> ops::Rem<I> for TDim {
    type Output = Self;
    fn rem(mut self, rhs: I) -> Self {
        self %= rhs;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! b( ($e:expr) => { Box::new($e) } );

    lazy_static::lazy_static! {
        static ref table: SymbolScope = SymbolScope::default();
        static ref A: Symbol = table.sym("a");
        static ref B: Symbol = table.sym("b");
    }

    fn neg(a: &TDim) -> TDim {
        mul(-1, a)
    }

    fn add(a: &TDim, b: &TDim) -> TDim {
        TDim::Add(vec![a.clone(), b.clone()])
    }

    fn mul(a: i64, b: &TDim) -> TDim {
        TDim::MulInt(a, b![b.clone()])
    }

    fn div(a: &TDim, b: u64) -> TDim {
        TDim::Div(b!(a.clone()), b)
    }

    #[test]
    fn reduce_add() {
        assert_eq!(add(&A.to_dim(), &neg(&A.to_dim())).reduce(), Val(0))
    }

    #[test]
    fn reduce_neg_mul() {
        assert_eq!(neg(&mul(2, &A.to_dim())).reduce(), mul(-2, &A.to_dim()))
    }

    #[test]
    fn reduce_cplx_ex_2() {
        assert_eq!(
            add(
                &add(&Val(-4), &mul(-2, &div(&A.to_dim(), 4))),
                &mul(-2, &mul(-1, &div(&A.to_dim(), 4)))
            )
            .reduce(),
            Val(-4)
        )
    }

    #[test]
    fn reduce_cplx_ex_3() {
        assert_eq!(div(&MulInt(1, b!(MulInt(4, b!(A.to_dim())))), 4).reduce(), A.to_dim())
    }

    #[test]
    fn reduce_cplx_ex_4() {
        // (S+1)/2 + (1-S)/2 == 1
        assert_eq!(
            add(&div(&add(&A.to_dim(), &Val(1)), 2), &div(&add(&neg(&A.to_dim()), &Val(1)), 2))
                .reduce(),
            1.into()
        );
    }

    #[test]
    fn reduce_mul_mul_1() {
        assert_eq!(mul(3, &mul(2, &A.to_dim())).reduce(), mul(6, &A.to_dim()))
    }

    #[test]
    fn reduce_mul_mul_2() {
        assert_eq!(mul(-2, &mul(-1, &A.to_dim())).reduce(), mul(2, &A.to_dim()))
    }

    #[test]
    fn reduce_mul_div_1() {
        assert_eq!(mul(2, &div(&mul(-1, &A.to_dim()), 3)).reduce(), mul(-2, &div(&A.to_dim(), 3)))
    }

    #[test]
    fn const_and_add() {
        let e: TDim = 2i64.into();
        assert_eq!(e.eval(&SymbolValues::default()).to_i64().unwrap(), 2);
        let e: TDim = TDim::from(2) + 3;
        assert_eq!(e.eval(&SymbolValues::default()).to_i64().unwrap(), 5);
        let e: TDim = TDim::from(2) - 3;
        assert_eq!(e.eval(&SymbolValues::default()).to_i64().unwrap(), -1);
        let e: TDim = -TDim::from(2);
        assert_eq!(e.eval(&SymbolValues::default()).to_i64().unwrap(), -2);
    }

    #[test]
    fn substitution() {
        let a: TDim = A.to_dim();
        assert_eq!(a.eval(&SymbolValues::default().with(&A, 2)).to_i64().unwrap(), 2);
        let e = a + 3;
        assert_eq!(e.eval(&SymbolValues::default().with(&A, 2)).to_i64().unwrap(), 5);
    }

    #[test]
    fn reduce_adds() {
        let e: TDim = TDim::from(2) + 1;
        assert_eq!(e, TDim::from(3));
        let e: TDim = TDim::from(3) + 2;
        assert_eq!(e, TDim::from(5));
        let e: TDim = TDim::from(3) + 0;
        assert_eq!(e, TDim::from(3));
        let e: TDim = TDim::from(3) + 2 + 1;
        assert_eq!(e, TDim::from(6));
    }

    #[test]
    fn reduce_muls() {
        let e: TDim = Val(1) * A.to_dim();
        assert_eq!(e, A.to_dim());
        let e: TDim = A.to_dim() * &B.to_dim() * 1;
        assert_eq!(e, A.to_dim() * &B.to_dim());
    }

    #[test]
    fn reduce_divs() {
        let e: TDim = TDim::from(2) / 1;
        assert_eq!(e, TDim::from(2));
        let e: TDim = TDim::from(3) / 2;
        assert_eq!(e, TDim::from(1));
        let e: TDim = TDim::from(3) % 2;
        assert_eq!(e, TDim::from(1));
        let e: TDim = TDim::from(5) / 2;
        assert_eq!(e, TDim::from(2));
        let e: TDim = TDim::from(5) % 2;
        assert_eq!(e, TDim::from(1));
    }

    #[test]
    fn reduce_div_bug_0() {
        let e1: TDim = (A.to_dim() + 23) / 2 - 1;
        let e2: TDim = (A.to_dim() + 21) / 2;
        assert_eq!(e1, e2);
    }

    #[test]
    fn reduce_div_bug_1() {
        let e1: TDim = (A.to_dim() + -1) / 2;
        let e2: TDim = (A.to_dim() + 1) / 2 - 1;
        assert_eq!(e1, e2);
    }

    #[test]
    fn reduce_div_bug_2() {
        let e1: TDim = ((A.to_dim() + 1) / 2 + 1) / 2;
        let e2: TDim = (A.to_dim() + 3) / 4;
        assert_eq!(e1, e2);
    }

    #[test]
    fn reduce_div_bug_3() {
        let e1: TDim = (A.to_dim() / 2) * -4;
        let e2: TDim = (A.to_dim() / 2) * -4 / 1;
        assert_eq!(e1, e2);
    }

    #[test]
    fn reduce_mul_div() {
        let e: TDim = A.to_dim() * 2 / 2;
        assert_eq!(e, A.to_dim());
    }

    #[test]
    fn reduce_div_mul() {
        let e: TDim = A.to_dim() / 2 * 2;
        assert_ne!(e, A.to_dim());
    }

    #[test]
    fn reduce_add_div() {
        let e: TDim = A.to_dim() / 2 + 1;
        assert_eq!(e, ((A.to_dim() + 2) / 2));
    }

    #[test]
    fn reduce_neg_mul_() {
        let e: TDim = TDim::from(1) - A.to_dim() * 2;
        assert_eq!(e, TDim::from(1) + A.to_dim() * -2);
    }

    #[test]
    fn reduce_add_rem_1() {
        assert_eq!(((A.to_dim() + 4) % 2), (A.to_dim() % 2));
    }

    #[test]
    fn reduce_add_rem_2() {
        assert_eq!(((A.to_dim() - 4) % 2), (A.to_dim() % 2));
    }

    #[test]
    fn reduce_rem_div() {
        let e: TDim = A.to_dim() % 2 / 2;
        assert_eq!(e, TDim::from(0));
    }

    #[test]
    fn conv2d_ex_1() {
        let e = (TDim::from(1) - 1 + 1).div_ceil(1);
        assert_eq!(e, TDim::from(1));
    }

    #[test]
    fn conv2d_ex_2() {
        let e = (A.to_dim() - 3 + 1).div_ceil(1);
        assert_eq!(e, A.to_dim() + -2);
    }

    #[test]
    fn extract_int_gcd_from_muls() {
        let term = (A.to_dim() + 1) / 4;
        let mul = (term.clone() * 24 - 24) * (term.clone() * 2 - 2);
        let target = (term.clone() - 1) * (term.clone() - 1) * 48;
        assert_eq!(mul, target);
    }

    #[test]
    fn equality_of_muls() {
        let term = (A.to_dim() + 1) / 4;
        let mul1 = (term.clone() * 2 - 3) * (term.clone() - 1);
        let mul2 = (term.clone() - 1) * (term.clone() * 2 - 3);
        assert_eq!(mul1, mul2);
    }

    #[test]
    fn factorize_complex_expr_times_int() {
        let term = (A.to_dim() + 1) / 4;
        let e = term.clone() * 2 - &term - 1;
        assert_eq!(e, term - 1);
    }

    #[test]
    fn min_ints_1() {
        assert_eq!(2.to_dim().mini(1.to_dim()), 1.to_dim());
    }

    #[test]
    fn min_ints_2() {
        assert_eq!(1.to_dim().mini(2.to_dim()), 1.to_dim());
    }

    #[test]
    fn min_same() {
        assert_eq!(A.to_dim().mini(A.to_dim()), A.to_dim());
    }

    #[test]
    fn min_noop() {
        assert_eq!(A.to_dim().mini(1.to_dim()), A.to_dim().mini(1.to_dim()));
    }

    #[test]
    fn min_diff_1() {
        assert_eq!((A.to_dim() + 1).mini(A.to_dim() + 2), A.to_dim() + 1);
    }

    #[test]
    fn slope_0() {
        assert_eq!(12.to_dim().guess_slope(&A), (0, 1));
    }

    #[test]
    fn slope_1() {
        assert_eq!(A.to_dim().guess_slope(&A), (1, 1));
    }

    #[test]
    fn slope_2() {
        assert_eq!((A.to_dim() * 2).guess_slope(&A), (2, 1));
    }

    #[test]
    fn slope_3() {
        assert_eq!((A.to_dim() * 2 + A.to_dim() / 2).guess_slope(&A), (5, 2));
    }

    #[test]
    fn slope_4() {
        assert_eq!((A.to_dim()).guess_slope(&B), (0, 1));
    }

    #[test]
    fn slope_5() {
        assert_eq!((A.to_dim() + 1).guess_slope(&A), (1, 1));
        assert_eq!((A.to_dim() + 1).guess_slope(&B), (0, 1));
    }

    #[test]
    fn slope_6() {
        assert_eq!((A.to_dim() + 1).guess_slope(&A), (1, 1));
        assert_eq!((A.to_dim() + B.to_dim()).guess_slope(&B), (1, 1));
    }

    #[test]
    fn min_0() -> TractResult<()> {
        let symbols = SymbolScope::default();
        assert_eq!(
            symbols.parse_tdim("min(S+3, S+2)").unwrap().simplify(),
            symbols.parse_tdim("S+2").unwrap(),
        );
        Ok(())
    }
}
