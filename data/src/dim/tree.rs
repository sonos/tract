use crate::dim::Assertion;
use crate::internal::*;

use super::{DimLike, sym::*};
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
    UndeterminedSymbol(String),
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
    /// Comparison: evaluates to 1 (true) or 0 (false). lhs >= rhs
    Ge(Box<TDim>, Box<TDim>),
    /// Comparison: evaluates to 1 (true) or 0 (false). lhs == rhs
    Eq(Box<TDim>, Box<TDim>),
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
        (Ge(a1, b1), Ge(a2, b2)) | (Eq(a1, b1), Eq(a2, b2)) => {
            tdim_lexi_order(a1, a2).then_with(|| tdim_lexi_order(b1, b2))
        }
        (Ge(_, _) | Eq(_, _), _) => Ordering::Less,
        (_, Ge(_, _) | Eq(_, _)) => Ordering::Greater,
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
            Ge(a, b) => write!(fmt, "({a}>={b})"),
            Eq(a, b) => write!(fmt, "({a}=={b})"),
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
            Err(TooEarly::UndeterminedSymbol(self.to_string()))?
        }
    }

    #[inline]
    pub fn as_i64(&self) -> Option<i64> {
        if let Val(v) = self { Some(*v) } else { None }
    }

    pub fn eval_to_i64(&self, values: &SymbolValues) -> TractResult<i64> {
        match self {
            Sym(sym) => {
                let Some(v) = values.get(sym) else {
                    Err(TooEarly::UndeterminedSymbol(self.to_string()))?
                };
                Ok(v)
            }
            Val(v) => Ok(*v),
            Add(terms) => terms.iter().try_fold(0i64, |acc, it| {
                let x = it.eval_to_i64(values)?;
                acc.checked_add(x)
                    .with_context(|| format!("Overflow in TDim addition ({acc} + {x})"))
            }),
            Mul(terms) => terms.iter().try_fold(1i64, |acc, it| {
                let x = it.eval_to_i64(values)?;
                acc.checked_mul(x)
                    .with_context(|| format!("Overflow in TDim multiplication ({acc} * {x})"))
            }),
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
            MulInt(p, a) => {
                let x = a.eval_to_i64(values)?;
                x.checked_mul(*p)
                    .with_context(|| format!("Overflow in TDim multiplication ({x} * {p})"))
            }
            Ge(a, b) => Ok(if a.eval_to_i64(values)? >= b.eval_to_i64(values)? { 1 } else { 0 }),
            Eq(a, b) => Ok(if a.eval_to_i64(values)? == b.eval_to_i64(values)? { 1 } else { 0 }),
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
            Ge(a, b) => {
                let a2 = a.eval(values);
                let b2 = b.eval(values);
                if let (Val(av), Val(bv)) = (&a2, &b2) {
                    Val(if av >= bv { 1 } else { 0 })
                } else {
                    Ge(b!(a2), b!(b2))
                }
            }
            Eq(a, b) => {
                let a2 = a.eval(values);
                let b2 = b.eval(values);
                if let (Val(av), Val(bv)) = (&a2, &b2) {
                    Val(if av == bv { 1 } else { 0 })
                } else {
                    Eq(b!(a2), b!(b2))
                }
            }
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
        self.clone().simplify_rec(&scope, Some(scenario), &[])
    }

    pub fn substitute(&self, from: &Symbol, to: &Self) -> TractResult<Self> {
        self.substitute_all(&std::collections::HashMap::from([(from.clone(), to.clone())]))
    }

    pub fn substitute_all(
        &self,
        map: &std::collections::HashMap<Symbol, Self>,
    ) -> TractResult<Self> {
        match self {
            Sym(sym) => Ok(map.get(sym).cloned().unwrap_or_else(|| self.clone())),
            Val(v) => Ok(Val(*v)),
            Add(terms) => terms.iter().try_fold(Val(0), |acc, it| -> TractResult<TDim> {
                Ok(acc + it.substitute_all(map)?)
            }),
            Mul(terms) => terms.iter().try_fold(Val(1), |acc, it| -> TractResult<TDim> {
                Ok(acc * it.substitute_all(map)?)
            }),
            Broadcast(terms) => terms.iter().try_fold(Val(1), |acc, it| -> TractResult<TDim> {
                acc.broadcast(it.substitute_all(map)?)
            }),
            Min(terms) => terms.iter().try_fold(Val(i64::MAX), |acc, it| -> TractResult<TDim> {
                Ok(acc.mini(it.substitute_all(map)?))
            }),
            Max(terms) => terms.iter().try_fold(Val(i64::MIN), |acc, it| -> TractResult<TDim> {
                Ok(acc.maxi(it.substitute_all(map)?))
            }),
            Div(a, q) => Ok(a.substitute_all(map)? / *q as i64),
            MulInt(p, a) => Ok(a.substitute_all(map)? * *p),
            Ge(a, b) => Ok(Ge(b!(a.substitute_all(map)?), b!(b.substitute_all(map)?))),
            Eq(a, b) => Ok(Eq(b!(a.substitute_all(map)?), b!(b.substitute_all(map)?))),
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
            Ge(a, b) | Eq(a, b) => 5 * (a.cost() + b.cost()),
        }
    }

    fn wiggle(&self) -> Vec<TDim> {
        use self::TDim::*;
        match self {
            Sym(_) | Val(_) | Mul(_) | Broadcast(_) | Min(_) | Max(_) | Ge(_, _) | Eq(_, _) => {
                vec![self.clone()]
            }
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
            Ge(a, b) | Eq(a, b) => Self::find_any_sym(a).or_else(|| Self::find_any_sym(b)),
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
        let it = self.simplify_rec(&scope, None, &[]);
        let mut current: Option<TDim> = None;
        for scenario in scope.scenarios() {
            let v = it.clone().simplify_rec(&scope, Some(scenario), &[]);
            if current.is_some_and(|c| c != v) {
                return it;
            } else {
                current = Some(v);
            }
        }
        current.unwrap_or(it)
    }

    pub fn simplify_with_extra_assertions(self, extra: &[Assertion]) -> TDim {
        use self::TDim::*;
        if extra.is_empty() {
            return self.simplify();
        }
        if let Ok(v) = self.eval_to_i64(&SymbolValues::default()) {
            return Val(v);
        }
        let Some(scope) = self.find_scope() else {
            return self;
        };
        let scope = scope.0;
        let locked = scope.lock();
        let scope = locked.borrow();
        let it = self.simplify_rec(&scope, None, extra);
        let mut current: Option<TDim> = None;
        for scenario in scope.scenarios() {
            let v = it.clone().simplify_rec(&scope, Some(scenario), extra);
            if current.is_some_and(|c| c != v) {
                return it;
            } else {
                current = Some(v);
            }
        }
        current.unwrap_or(it)
    }

    fn simplify_rec(
        self,
        scope: &SymbolScopeData,
        scenario: Option<&str>,
        extra: &[Assertion],
    ) -> TDim {
        match self {
            Add(mut terms) => {
                #[allow(clippy::mutable_key_type)]
                let mut simplified_terms: HashMap<TDim, i64> = HashMap::new();
                // factorize common sub-expr
                while let Some(term) = terms.pop() {
                    let simplified = term.simplify_rec(scope, scenario, extra);
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
                // Distribute over Add: if exactly one factor is an Add,
                // expand Mul([a, Add([b, c])]) => Add([Mul([a, b]), Mul([a, c])]).
                // This lets (T+1)*P simplify to T*P + P, which is needed for
                // cancellation in expressions like (T+1)*P - T*P.
                {
                    let add_indices: Vec<usize> = terms
                        .iter()
                        .enumerate()
                        .filter(|(_, t)| matches!(t, Add(_)))
                        .map(|(i, _)| i)
                        .collect();
                    if add_indices.len() == 1 {
                        let add_idx = add_indices[0];
                        let Add(add_terms) = &terms[add_idx] else { unreachable!() };
                        let other_factors: Vec<TDim> = terms
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i != add_idx)
                            .map(|(_, t)| t.clone())
                            .collect();
                        let distributed: Vec<TDim> = add_terms
                            .iter()
                            .map(|at| {
                                let mut product = other_factors.clone();
                                product.push(at.clone());
                                Mul(product)
                            })
                            .collect();
                        return Add(distributed).simplify_rec(scope, scenario, extra);
                    }
                }

                // in case a term is a multiplication itself, flatten it
                // e.g., (a*b)*c => a*b*c, and MulInt(k, x) => Val(k)*x
                let mut flattened_terms = vec![];
                for t in terms {
                    match t.clone().reduce() {
                        Mul(inner_terms) => flattened_terms.extend(inner_terms),
                        MulInt(k, inner) => {
                            flattened_terms.push(Val(k));
                            flattened_terms.push(*inner);
                        }
                        other => flattened_terms.push(other),
                    }
                }
                let mut terms = flattened_terms;

                let mut gcd = Mul(terms.clone()).gcd() as i64;
                if gcd == 0 {
                    return Val(0);
                }
                terms = if gcd != 1 {
                    terms
                        .into_iter()
                        .map(|t| {
                            let gcd = t.gcd();
                            (t / gcd).simplify_rec(scope, scenario, extra)
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
                        if let Some(c) = coef.checked_mul(c2) {
                            return MulInt(c, inner).simplify_rec(scope, scenario, extra);
                        } else {
                            return MulInt(coef, Box::new(MulInt(c2, inner)));
                        }
                    }
                    Val(v) => {
                        return coef
                            .checked_mul(v)
                            .map(Val)
                            .unwrap_or_else(|| MulInt(coef, Box::new(Val(v))));
                    }
                    _ => {}
                }

                let simplified = expr.simplify_rec(scope, scenario, extra);
                match (coef, simplified) {
                    (0, _) => Val(0), // Case #1: If coef is 0, return 0
                    (1, s) => s,      // Case #2: If coef is 1, return the simplified expression
                    (_, Add(terms)) => Add(terms
                        .into_iter()
                        .map(|term| {
                            MulInt(coef, Box::new(term)).simplify_rec(scope, scenario, extra)
                        })
                        .collect()), // Case #3: If expression is an addition, distribute the coef
                    (c, Val(v)) => {
                        c.checked_mul(v).map(Val).unwrap_or_else(|| MulInt(c, Box::new(Val(v))))
                    } // Case #4: If expression is a value, combine coefs
                    (c, MulInt(v, inner)) => {
                        if let Some(cv) = c.checked_mul(v) {
                            MulInt(cv, inner) // Case #5: If expression is a MulInt, combine coefs
                        } else {
                            MulInt(c, Box::new(MulInt(v, inner)))
                        }
                    }
                    (_, s) => MulInt(coef, Box::new(s)), // Case #6: Otherwise, return the original
                }
            }
            Div(a, q) => {
                if q == 1 {
                    return a.simplify_rec(scope, scenario, extra);
                } else if let Div(a, q2) = *a {
                    return Div(a, q * q2).simplify_rec(scope, scenario, extra);
                }
                let a = a.simplify_rec(scope, scenario, extra);
                if let Val(a) = a {
                    Val(a / q as i64)
                } else if let MulInt(-1, a) = a {
                    MulInt(-1, b!(Div(a, q)))
                } else if let Add(mut terms) = a {
                    if terms
                        .iter()
                        .any(|t| if let MulInt(-1, s) = t { matches!(&**s, Sym(_)) } else { false })
                    {
                        MulInt(
                            -1,
                            b!(Div(
                                b!(Add(terms.into_iter().map(|t| MulInt(-1, b!(t))).collect())
                                    .simplify_rec(scope, scenario, extra)),
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
                                Div(b!(Add(terms).simplify_rec(scope, scenario, extra)), q),
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
                                .simplify_rec(scope, scenario, extra)
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
                    .map(|s| s.clone().simplify_rec(scope, scenario, extra))
                    .flat_map(|t| if let Broadcast(t) = t { t } else { vec![t] })
                    .filter(|t| !t.is_one())
                    .sorted_by(tdim_lexi_order)
                    .dedup()
                    .collect_vec();
                // a#min(a,b) if a>0 && b>0 => a
                match &*terms {
                    [] => Val(1),
                    [_] => terms.remove(0),
                    [a, Min(m)] | [Min(m), a]
                        if m.contains(a)
                            && m.iter()
                                .all(|t| scope.prove_strict_positive_with_extra(t, extra)) =>
                    {
                        a.clone()
                    }
                    _ => Broadcast(terms),
                }
            }

            Min(terms) => {
                let mut flatten: Vec<TDim> = terms
                    .into_iter()
                    .map(|t| t.simplify_rec(scope, scenario, extra))
                    .flat_map(|t| if let Min(t) = t { t } else { vec![t] })
                    .filter(|t| t != &Val(i64::MAX))
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
                    if diff.as_i64().is_some_and(|i| i >= 0)
                        || scope.prove_positive_or_zero_with_extra(&diff, extra)
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
                    .map(|t| t.simplify_rec(scope, scenario, extra))
                    .flat_map(|t| if let Max(t) = t { t } else { vec![t] })
                    .filter(|t| t != &Val(i64::MIN))
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
                    if diff.as_i64().is_some_and(|i| i >= 0)
                        || scope.prove_positive_or_zero_with_extra(&diff, extra)
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
            Ge(a, b) => {
                let a = a.simplify_rec(scope, scenario, extra);
                let b = b.simplify_rec(scope, scenario, extra);
                match (&a, &b) {
                    (Val(av), Val(bv)) => Val(if av >= bv { 1 } else { 0 }),
                    _ => {
                        let diff = a.clone() - b.clone();
                        if scope.prove_positive_or_zero_with_extra(&diff, extra) {
                            Val(1)
                        } else if scope
                            .prove_strict_positive_with_extra(&(b.clone() - a.clone()), extra)
                        {
                            Val(0)
                        } else {
                            Ge(b!(a), b!(b))
                        }
                    }
                }
            }
            Eq(a, b) => {
                let a = a.simplify_rec(scope, scenario, extra);
                let b = b.simplify_rec(scope, scenario, extra);
                match (&a, &b) {
                    (Val(av), Val(bv)) => Val(if av == bv { 1 } else { 0 }),
                    _ => {
                        let diff = a.clone() - b.clone();
                        if scope.prove_strict_positive_with_extra(&diff, extra)
                            || scope
                                .prove_strict_positive_with_extra(&(b.clone() - a.clone()), extra)
                        {
                            Val(0)
                        } else {
                            // When one side is 0 or 1 and the other is
                            // provably in [0,1], reduce to boolean algebra:
                            //   Eq(expr, 0) → 1 - expr
                            //   Eq(expr, 1) → expr
                            let boolean_case = match (&a, &b) {
                                (Val(0), e) | (e, Val(0)) => Some((e, false)),
                                (Val(1), e) | (e, Val(1)) => Some((e, true)),
                                _ => None,
                            };
                            if let Some((expr, equals_one)) = boolean_case {
                                if scope.prove_positive_or_zero_with_extra(expr, extra)
                                    && scope.prove_positive_or_zero_with_extra(
                                        &(Val(1) - expr.clone()),
                                        extra,
                                    )
                                {
                                    return if equals_one {
                                        expr.clone()
                                    } else {
                                        (Val(1) - expr.clone()).simplify_rec(scope, scenario, extra)
                                    };
                                }
                            }
                            Eq(b!(a), b!(b))
                        }
                    }
                }
            }
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
                let mut bound: i64 = 0;
                for t in terms {
                    if let Some(b) = t.inclusive_bound(scope, upper) {
                        bound = bound.checked_add(b)?;
                    } else {
                        return None;
                    }
                }
                Some(bound)
            }
            MulInt(p, a) => match p.cmp(&0) {
                Ordering::Equal => Some(0),
                Ordering::Greater => {
                    a.inclusive_bound(scope, upper).and_then(|x| x.checked_mul(*p))
                }
                Ordering::Less => a.inclusive_bound(scope, !upper).and_then(|x| x.checked_mul(*p)),
            },
            Mul(terms) => {
                // If all factors have known non-negative bounds, we can bound the product.
                let mut lo: i64 = 1;
                let mut hi: i64 = 1;
                for t in terms {
                    let t_lo = t.inclusive_bound(scope, false)?;
                    let t_hi = t.inclusive_bound(scope, true)?;
                    if t_lo < 0 {
                        return None;
                    }
                    lo = lo.checked_mul(t_lo)?;
                    hi = hi.checked_mul(t_hi)?;
                }
                Some(if upper { hi } else { lo })
            }
            Min(terms) if !upper => {
                // All terms must have known lower bounds; if any is unknown,
                // the Min lower bound is unknown.
                let bounds: Option<Vec<i64>> =
                    terms.iter().map(|t| t.inclusive_bound(scope, false)).collect();
                bounds.map(|b| b.into_iter().min().unwrap_or(i64::MAX))
            }
            Max(terms) if upper => {
                // All terms must have known upper bounds; if any is unknown,
                // the Max upper bound is unknown.
                let bounds: Option<Vec<i64>> =
                    terms.iter().map(|t| t.inclusive_bound(scope, true)).collect();
                bounds.map(|b| b.into_iter().max().unwrap_or(i64::MIN))
            }
            Div(a, q) => a.inclusive_bound(scope, upper).map(|x| x / (*q as i64)),
            Broadcast(terms) => {
                if upper {
                    Max(terms.clone()).inclusive_bound(scope, true)
                } else {
                    Min(terms.clone()).inclusive_bound(scope, false)
                }
            }
            Ge(_, _) | Eq(_, _) => {
                if upper {
                    Some(1)
                } else {
                    Some(0)
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
            MulInt(p, a) => a.gcd().saturating_mul(p.unsigned_abs()),
            Mul(terms) => terms.iter().map(|t| t.gcd()).fold(1u64, |a, b| a.saturating_mul(b)),
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
            Ge(_, _) | Eq(_, _) => 1,
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
            Ge(_, _) | Eq(_, _) => Div(Box::new(self.clone()), d),
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
                Ge(_, _) | Eq(_, _) => (0, 1),
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
            Ge(a, b) | Eq(a, b) => {
                let mut set = a.symbols();
                set.extend(b.symbols());
                set
            }
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
    if q < 0 { (-p, (-q) as u64) } else { (p, q as u64) }
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
        if let Val(v) = self { Val(-v) } else { TDim::MulInt(-1, Box::new(self)).reduce() }
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
        static ref C: Symbol = table.sym("c");
        static ref D: Symbol = table.sym("d");
        static ref E: Symbol = table.sym("e");
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
    fn broadcast_over_min() {
        // assuming a>0, b>0 then a#min(a,b) can be replaced by a
        // proof:
        //    if b == 1 => min(a,b)=1 => a#1=a => ok
        //    if a <= b => min(a,b)=a => ok
        //    if 1 < B < A => expression was invalid, we're generalizing over the non-domain and ignoring the constraint
        for a in 1..5 {
            for b in 1..5 {
                if b > 1 && a > b {
                    assert!(a.broadcast(a.min(b)).is_err());
                } else {
                    assert_eq!(a.broadcast(a.min(b)).unwrap(), a);
                }
            }
        }
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

    #[test]
    fn commutative_mul_parens() -> TractResult<()> {
        let symbols = SymbolScope::default();
        assert_eq!(
            symbols.parse_tdim("A*(B*C)").unwrap().simplify(),
            symbols.parse_tdim("(B*A)*C").unwrap().simplify(),
        );
        Ok(())
    }

    #[test]
    fn commutative_in_nemo_parakeet_model() -> TractResult<()> {
        let symbols = SymbolScope::default();
        assert_eq!(
            symbols
                .parse_tdim("8*(1+-1*max(0,5000+-1*(S+7)/8)+max(0,4999+(S+7)/8))*((B)*((S+7)/8))")
                .unwrap()
                .simplify(),
            symbols
                .parse_tdim("8*((B)*(1+-1*max(0,5000+-1*(S+7)/8)+max(0,4999+(S+7)/8)))*((S+7)/8)")
                .unwrap()
                .simplify(),
        );
        Ok(())
    }

    #[test]
    fn commutative_mul_parens_deep() -> TractResult<()> {
        let symbols = SymbolScope::default();
        let deep_tdim = Mul(vec![
            Mul(vec![Mul(vec![Mul(vec![A.to_dim(), B.to_dim()]), C.to_dim()]), D.to_dim()]),
            E.to_dim(),
        ])
        .simplify();
        assert_eq!(deep_tdim, symbols.parse_tdim("a*b*c*d*e").unwrap().simplify());
        Ok(())
    }

    // ---- Tests for new comparison/not TDim variants ----

    #[test]
    fn ge_concrete_true() {
        assert_eq!(Ge(b!(Val(5)), b!(Val(3))).reduce(), Val(1));
    }

    #[test]
    fn ge_concrete_false() {
        assert_eq!(Ge(b!(Val(2)), b!(Val(3))).reduce(), Val(0));
    }

    #[test]
    fn lt_concrete_true() {
        // Lt(2,3) normalizes to Ge(3, 2+1) = Ge(3, 3)
        assert_eq!(Ge(b!(Val(3)), b!(Val(3))).reduce(), Val(1));
    }

    #[test]
    fn lt_concrete_false() {
        // Lt(5,3) normalizes to Ge(3, 5+1) = Ge(3, 6)
        assert_eq!(Ge(b!(Val(3)), b!(Val(6))).reduce(), Val(0));
    }

    #[test]
    fn eq_concrete_true() {
        assert_eq!(Eq(b!(Val(3)), b!(Val(3))).reduce(), Val(1));
    }

    #[test]
    fn eq_concrete_false() {
        assert_eq!(Eq(b!(Val(3)), b!(Val(4))).reduce(), Val(0));
    }

    #[test]
    fn not_val_0() {
        // not(0) = 1 - 0 = 1
        assert_eq!((Val(1) - Val(0)).reduce(), Val(1));
    }

    #[test]
    fn not_val_1() {
        // not(1) = 1 - 1 = 0
        assert_eq!((Val(1) - Val(1)).reduce(), Val(0));
    }

    #[test]
    fn not_lt_becomes_ge() {
        // not(Lt(x1, T)) = 1 - Ge(T, x1+1); check it evaluates correctly at boundary
        let s = SymbolScope::default();
        let t = s.sym("T");
        let x1 = s.sym("x1");
        // at x1 = T (boundary), Ge(T, T+1) = 0, so 1 - 0 = 1 (not-lt is true when x1 >= T)
        let expr = Val(1) - Ge(b!(Sym(t.clone())), b!(Sym(x1.clone()) + Val(1)));
        let at_boundary = expr.substitute(&x1, &Sym(t.clone())).unwrap().simplify();
        assert_eq!(at_boundary, Val(1));
    }

    #[test]
    fn eq_with_assertion_proves_false() {
        // Eq(T, 0) should reduce to Val(0) when T >= 1
        let s = SymbolScope::default();
        s.add_assertion("T >= 1").unwrap();
        let t = s.sym("T");
        let expr = Eq(b!(Sym(t)), b!(Val(0)));
        assert_eq!(expr.simplify(), Val(0));
    }

    #[test]
    fn ge_coord_at_extremes() {
        // Ge(x1, T) should not simplify without coordinate substitution
        let s = SymbolScope::default();
        s.add_assertion("T >= 1").unwrap();
        let t = s.sym("T");
        let x1 = s.sym("x1");
        let expr = Ge(b!(Sym(x1.clone())), b!(Sym(t.clone())));
        // simplify() alone can't prove this false (x1 could be > T)
        // but with coordinate substitution (x1 = T-1), Ge(T-1, T) = 0
        let at_max = expr.substitute(&x1, &(Sym(t.clone()) - Val(1))).unwrap().simplify();
        assert_eq!(at_max, Val(0));
    }

    #[test]
    fn eval_to_i64_new_variants() {
        use super::super::sym::SymbolValues;
        let sv = SymbolValues::default();
        assert_eq!(Ge(b!(Val(5)), b!(Val(3))).eval_to_i64(&sv).unwrap(), 1);
        assert_eq!(Ge(b!(Val(3)), b!(Val(5))).eval_to_i64(&sv).unwrap(), 0);
        assert_eq!(Eq(b!(Val(3)), b!(Val(3))).eval_to_i64(&sv).unwrap(), 1);
        assert_eq!(Eq(b!(Val(3)), b!(Val(4))).eval_to_i64(&sv).unwrap(), 0);
    }

    #[test]
    fn eq_boolean_simplifies() {
        let s = SymbolScope::default();
        s.add_assertion("cw >= 0").unwrap();
        s.add_assertion("cw <= 1").unwrap();
        let cw = s.sym("cw");
        // Eq(1 - cw, 0) → cw
        assert_eq!(Eq(b!(Val(1) - Sym(cw.clone())), b!(Val(0))).simplify(), Sym(cw.clone()));
        // Eq(cw, 0) → 1 - cw
        assert_eq!(Eq(b!(Sym(cw.clone())), b!(Val(0))).simplify(), Val(1) - Sym(cw.clone()));
        // Eq(cw, 1) → cw
        assert_eq!(Eq(b!(Sym(cw.clone())), b!(Val(1))).simplify(), Sym(cw.clone()));
        // Eq(1 - cw, 1) → 1 - cw
        assert_eq!(Eq(b!(Val(1) - Sym(cw.clone())), b!(Val(1))).simplify(), Val(1) - Sym(cw));
    }

    #[test]
    fn eq_boolean_mul_of_ge() {
        // Product of Ge terms: Ge(a,b) * Ge(c,d) is in [0,1]
        // so Eq(product, 0) should simplify to 1 - product
        let s = SymbolScope::default();
        let x = s.sym("x");
        let product =
            Mul(vec![Ge(b!(Val(2)), b!(Sym(x.clone()))), Ge(b!(Sym(x.clone())), b!(Val(0)))]);
        let eq = Eq(b!(product.clone()), b!(Val(0)));
        assert_eq!(eq.simplify(), Val(1) - product);
    }

    #[test]
    fn min_1_max_0_sym() {
        // Min(1, Max(0, X)) must not simplify away the Min when X is unconstrained.
        let s = SymbolScope::default();
        let x = s.sym("X");
        let expr = Min(vec![Val(1), Max(vec![Val(0), Sym(x)])]);
        let simplified = expr.simplify();
        eprintln!("simplified: {simplified}");
        assert!(format!("{simplified}").contains("min"), "Min dropped: {simplified}");
    }

    #[test]
    fn min_preserved_in_subtraction_parts() {
        // Test that Min([1, X]) simplifies correctly in isolation
        let s = SymbolScope::default();
        let t = s.sym("T");
        let p = s.sym("P");
        let ss = s.sym("S");

        let cum_after =
            Max(vec![Val(0), (Sym(t.clone()) + Val(1)) * Sym(p.clone()) - Sym(ss.clone())]);
        let min_after = Min(vec![Val(1), cum_after.clone()]);
        let simplified = min_after.simplify();
        eprintln!("min_after simplified: {simplified}");
        // Must contain "min" — the Min must not be dropped
        assert!(format!("{simplified}").contains("min"), "Min wrapper was dropped: {simplified}");
    }

    #[test]
    fn min_preserved_in_subtraction() {
        // min(1, X) - min(1, Y) must preserve the min() wrappers.
        // This is the pattern used by PulseV2Pad's output_facts for after-padding.
        let s = SymbolScope::default();
        let t = s.sym("T");
        let p = s.sym("P");
        let ss = s.sym("S");

        let cum_after =
            Max(vec![Val(0), (Sym(t.clone()) + Val(1)) * Sym(p.clone()) - Sym(ss.clone())]);
        let cum_before = Max(vec![Val(0), Sym(t.clone()) * Sym(p.clone()) - Sym(ss.clone())]);

        let ap = Min(vec![Val(1), cum_after.clone()]) - Min(vec![Val(1), cum_before.clone()]);
        let simplified = ap.simplify();

        // At T=1, P=4, S=3: min(1, max(0, 8-3)) - min(1, max(0, 4-3)) = 1 - 1 = 0
        use super::super::sym::SymbolValues;
        let sv = SymbolValues::default().with(&t, 1).with(&p, 4).with(&ss, 3);
        assert_eq!(simplified.eval_to_i64(&sv).unwrap(), 0, "simplified: {simplified}");

        // At T=0, P=4, S=3: min(1, max(0, 4-3)) - min(1, max(0, 0-3)) = 1 - 0 = 1
        let sv = SymbolValues::default().with(&t, 0).with(&p, 4).with(&ss, 3);
        assert_eq!(simplified.eval_to_i64(&sv).unwrap(), 1, "simplified: {simplified}");

        // At T=0, P=1, S=1: min(1, max(0, 1-1)) - min(1, max(0, 0-1)) = 0 - 0 = 0
        let sv = SymbolValues::default().with(&t, 0).with(&p, 1).with(&ss, 1);
        assert_eq!(simplified.eval_to_i64(&sv).unwrap(), 0, "simplified: {simplified}");
    }

    #[test]
    fn mul_neg_b_by_8() {
        let s = SymbolScope::default();
        let b = Sym(s.sym("B"));
        // 8*(-1*B) should equal -8*B
        let a = Mul(vec![Val(8), MulInt(-1, Box::new(b.clone()))]);
        let c = MulInt(-8, Box::new(b.clone()));
        let a_s = a.simplify();
        let c_s = c.simplify();
        assert_eq!(a_s, c_s, "8*(-1*B) should simplify the same as -8*B");
    }
}

#[test]
fn mul_neg_b_by_8() {
    let s = crate::dim::SymbolScope::default();
    let b = Sym(s.sym("B"));
    // 8*(-1*B) should equal -8*B
    let a = Mul(vec![Val(8), MulInt(-1, Box::new(b.clone()))]);
    let c = MulInt(-8, Box::new(b.clone()));
    let a_s = a.simplify();
    let c_s = c.simplify();
    eprintln!("8*(-1*B) = {a_s}");
    eprintln!("-8*B     = {c_s}");
    assert_eq!(a_s, c_s, "8*(-1*B) should simplify the same as -8*B");
}
