use itertools::Itertools;
use num_traits::{AsPrimitive, PrimInt, Zero};
use std::collections::HashMap;
use std::{fmt, ops};

#[derive(Debug)]
pub struct UndeterminedSymbol(TDim);

impl std::fmt::Display for UndeterminedSymbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Undetermined symbol in expression: {}", self.0)
    }
}

impl std::error::Error for UndeterminedSymbol {}

macro_rules! b( ($e:expr) => { Box::new($e) } );

lazy_static::lazy_static! {
    static ref SYMBOL_TABLE: std::sync::Mutex<Vec<char>> = std::sync::Mutex::new(Vec::new());
}

#[derive(Copy, Clone, PartialEq, Eq, Ord, PartialOrd, Hash, Debug)]
pub struct Symbol(char, usize);

impl Symbol {
    pub fn new(c: char) -> Symbol {
        let mut table = SYMBOL_TABLE.lock().unwrap();
        table.push(c);
        Symbol(c, table.len() - 1)
    }

    pub fn as_char(&self) -> char {
        self.0
    }
}

impl From<char> for Symbol {
    fn from(c: char) -> Symbol {
        let mut table = SYMBOL_TABLE.lock().unwrap();
        if let Some(pos) = table.iter().position(|s| *s == c) {
            Symbol(c, pos)
        } else {
            table.push(c);
            Symbol(c, table.len() - 1)
        }
    }
}

impl From<char> for TDim {
    fn from(c: char) -> TDim {
        Symbol::from(c).into()
    }
}

#[derive(Clone, Debug, Default)]
pub struct SymbolValues(Vec<Option<i64>>);

impl SymbolValues {
    pub fn with(mut self, s: Symbol, v: i64) -> Self {
        self[s] = Some(v);
        self
    }
}

impl std::ops::Index<Symbol> for SymbolValues {
    type Output = Option<i64>;
    fn index(&self, index: Symbol) -> &Self::Output {
        if index.1 < self.0.len() {
            &self.0[index.1]
        } else {
            &None
        }
    }
}

impl std::ops::IndexMut<Symbol> for SymbolValues {
    fn index_mut(&mut self, index: Symbol) -> &mut Self::Output {
        if index.1 >= self.0.len() {
            self.0.resize_with(index.1 + 1, Default::default)
        }
        &mut self.0[index.1]
    }
}

#[derive(Clone, PartialEq, Eq, Ord, PartialOrd, Hash, Debug)]
pub enum TDim {
    Sym(Symbol),
    Val(i64),
    Add(Vec<TDim>),
    Mul(Vec<TDim>),
    MulInt(i64, Box<TDim>),
    Div(Box<TDim>, u64),
}

use TDim::*;

impl fmt::Display for TDim {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match &self {
            Sym(sym) => write!(fmt, "{}", sym.0),
            Val(it) => write!(fmt, "{}", it),
            Add(it) => write!(fmt, "{}", it.iter().map(|x| format!("{}", x)).join("+")),
            Mul(it) => write!(fmt, "{}", it.iter().map(|x| format!("{}", x)).join("*")),
            MulInt(a, b) => write!(fmt, "{}*{}", a, b),
            Div(a, b) => write!(fmt, "({})/{}", a, b),
        }
    }
}

impl TDim {
    pub fn is_one(&self) -> bool {
        self == &Val(1)
    }

    pub fn to_i64(&self) -> anyhow::Result<i64> {
        if let Val(v) = self {
            Ok(*v)
        } else {
            Err(UndeterminedSymbol(self.clone()).into())
        }
    }

    pub fn eval(&self, values: &SymbolValues) -> TDim {
        match self {
            Sym(sym) => values[*sym].map(|s| Val(s)).unwrap_or(Sym(*sym)),
            Val(v) => Val(*v),
            Add(terms) => terms.iter().fold(Val(0), |acc, it| -> TDim { acc + it.eval(values) }),
            Mul(terms) => terms.iter().fold(Val(1), |acc, it| -> TDim { acc * it.eval(values) }),
            Div(a, q) => a.eval(values) / *q as i64,
            MulInt(p, a) => a.eval(values) * *p,
        }
    }

    pub fn reduce(self) -> TDim {
        self.simplify()
            .wiggle()
            .into_iter()
            .sorted()
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
            Div(a, _) => 3 * a.cost(),
            MulInt(_, a) => 2 * a.cost(),
        }
    }

    fn wiggle(&self) -> Vec<TDim> {
        use self::TDim::*;
        match self {
            Sym(_) | Val(_) | Mul(_) => vec![self.clone()],
            Add(terms) => {
                let mut forms = vec![];
                let sub_wiggle = terms.iter().map(|e| e.wiggle()).multi_cartesian_product();
                for sub in sub_wiggle {
                    for (ix, num, q) in sub.iter().enumerate().find_map(|(ix, t)| {
                        if let Div(a, q) = t {
                            Some((ix, a, q))
                        } else {
                            None
                        }
                    }) {
                        let new_num = sub
                            .iter()
                            .enumerate()
                            .map(|(ix2, t)| {
                                if ix2 != ix {
                                    MulInt(*q as i64, b!(t.clone()))
                                } else {
                                    (**num).clone()
                                }
                            })
                            .collect();
                        forms.push(Div(b!(Add(new_num)), *q))
                    }
                    forms.push(Add(sub.into()));
                }
                forms
            }
            MulInt(p, a) => a.wiggle().into_iter().map(|a| MulInt(*p, b!(a))).collect(),
            Div(a, q) => {
                let mut forms = vec![];
                for num in a.wiggle() {
                    if let Add(terms) = &num {
                        let (integer, non_integer): (Vec<_>, Vec<_>) =
                            terms.into_iter().cloned().partition(|a| a.gcd() % q == 0);
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

    pub fn simplify(self) -> TDim {
        use self::TDim::*;
        use num_integer::Integer;
        match self {
            Add(mut terms) => {
                let mut reduced: HashMap<TDim, i64> = HashMap::new();
                // factorize common sub-expr
                while let Some(item) = terms.pop() {
                    let term = item.simplify();
                    match term {
                        Add(items) => {
                            terms.extend(items.into_iter());
                            continue;
                        }
                        Val(0) => (),
                        Val(v) => *reduced.entry(Val(1)).or_insert(0) += v,
                        MulInt(v, f) => {
                            *reduced.entry((*f).clone()).or_insert(0) += v;
                        }
                        n => *reduced.entry(n).or_insert(0) += 1,
                    };
                }
                let mut members: Vec<_> = reduced
                    .into_iter()
                    .filter_map(|(k, v)| {
                        if v == 0 {
                            None
                        } else if k == Val(1) {
                            Some(Val(v))
                        } else if v == 1 {
                            Some(k)
                        } else {
                            Some(MulInt(v, b![k]))
                        }
                    })
                    .collect();
                members.sort();
                if members.len() == 0 {
                    Val(0)
                } else if members.len() > 1 {
                    Add(members)
                } else {
                    members.remove(0)
                }
            }
            Mul(terms) => {
                let (ints, mut rest): (i64, Vec<TDim>) =
                    terms.into_iter().fold((1, vec![]), |acc, t| match t.simplify() {
                        MulInt(a, p) => {
                            (acc.0 * a, acc.1.into_iter().chain(Some(p.as_ref().clone())).collect())
                        }
                        Val(a) => (acc.0 * a, acc.1),
                        it => {
                            (acc.0, acc.1.into_iter().chain(Some(it.clone()).into_iter()).collect())
                        }
                    });
                if rest.len() == 0 {
                    Val(ints)
                } else if ints == 0 {
                    Val(0)
                } else {
                    let rest = if rest.len() == 1 { rest.remove(0) } else { Mul(rest) };
                    if ints == 1 {
                        rest
                    } else {
                        MulInt(ints, Box::new(rest))
                    }
                }
            }
            MulInt(p, a) => {
                if let MulInt(p2, a) = *a {
                    return MulInt(p * p2, a).simplify();
                } else if let Val(p2) = *a {
                    return Val(p * p2);
                }
                let a = a.simplify();
                if p == 0 {
                    Val(0)
                } else if p == 1 {
                    a
                } else if let Add(terms) = &a {
                    Add(terms.clone().into_iter().map(|a| MulInt(p, b!(a)).simplify()).collect())
                } else if let Val(p2) = a {
                    Val(p * p2)
                } else if let MulInt(p2, a) = a {
                    MulInt(p * p2, a)
                } else {
                    MulInt(p, b!(a))
                }
            }
            Div(a, q) => {
                if q == 1 {
                    return a.simplify();
                } else if let Div(a, q2) = *a {
                    return Div(a, q * q2).simplify();
                }
                let a = a.simplify();
                if let Val(a) = a {
                    Val(a / q as i64)
                } else if let MulInt(-1, a) = a {
                    MulInt(-1, b!(Div(a, q)))
                } else if let Add(mut terms) = a {
                    if terms.iter().any(|t| {
                        if let MulInt(-1, s) = t {
                            if let Sym(_) = &**s {
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    }) {
                        MulInt(
                            -1,
                            b!(Div(
                                b!(Add(terms.into_iter().map(|t| MulInt(-1, b!(t))).collect())
                                    .simplify()),
                                q
                            )),
                        )
                    } else if let Some(v) = terms
                        .iter()
                        .filter_map(|t| if let Val(v) = t { Some(*v) } else { None })
                        .next()
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
                            Add(vec![Val(val), Div(b!(Add(terms).simplify()), q)])
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
                            Div(b!(MulInt(p / gcd, a)), q / gcd as u64).simplify()
                        } else {
                            Div(b!(MulInt(p, a)), q)
                        }
                    }
                } else {
                    Div(b!(a), q)
                }
            }
            _ => self,
        }
    }

    fn gcd(&self) -> u64 {
        use self::TDim::*;
        use num_integer::Integer;
        match self {
            Val(v) => v.abs() as u64,
            Sym(_) => 1,
            Add(terms) => {
                let (head, tail) = terms.split_first().unwrap();
                tail.iter().fold(head.gcd(), |a, b| a.gcd(&b.gcd()))
            }
            MulInt(p, a) => a.gcd() * p.abs() as u64,
            Mul(_) => 1,
            Div(a, q) => {
                if a.gcd() % *q == 0 {
                    a.gcd() / *q
                } else {
                    1
                }
            }
        }
    }

    fn div(&self, d: u64) -> TDim {
        use self::TDim::*;
        use num_integer::Integer;
        if d == 1 {
            return self.clone();
        }
        match self {
            Val(v) => Val(v / d as i64),
            Sym(_) => panic!(),
            Add(terms) => Add(terms.iter().map(|t| t.div(d)).collect()),
            Mul(_) => Div(Box::new(self.clone()), d),
            MulInt(p, a) => {
                if *p == d as i64 {
                    (**a).clone()
                } else {
                    let gcd = (p.abs() as u64).gcd(&d);
                    MulInt(p / gcd as i64, b!(a.div(d / gcd)))
                }
            }
            Div(a, q) => Div(a.clone(), q * d),
        }
    }

    pub fn div_ceil(self, rhs: u64) -> TDim {
        TDim::Div(Box::new(Add(vec![self, Val(rhs as i64 - 1)])), rhs).reduce()
    }

    pub fn slope(&self, sym: Symbol) -> (i64, u64) {
        fn slope_rec(d: &TDim, sym: Symbol) -> (i64, i64) {
            match d {
                Val(_) => (0, 1),
                Sym(s) => ((sym == *s) as i64, 1),
                Add(terms) => terms
                    .iter()
                    .map(|d| slope_rec(d, sym))
                    .fold((1, 1), |a, b| ((a.0 * b.1 + a.1 * b.0), (b.1 * a.1))),
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
            }
        }
        let (p, q) = slope_rec(self, sym);
        reduce_ratio(p, q)
    }

    pub fn symbols(&self) -> std::collections::HashSet<Symbol> {
        match self {
            Val(_) => maplit::hashset!(),
            Sym(s) => maplit::hashset!(*s),
            Add(terms) | Mul(terms) => terms.iter().fold(maplit::hashset!(), |mut set, v| {
                set.extend(v.symbols().into_iter());
                set
            }),
            MulInt(_, a) => a.symbols(),
            Div(a, _) => a.symbols(),
        }
    }

    /// Check if a dim is 'compatible with' another, meaning that the current dim
    /// is a "sub" dimension within or equal to the _other dim
    pub fn compatible_with(&self, _other: &TDim) -> bool {
        match (self, _other) {
            // If we compare a concrete dim to symbol dim we are always
            // true but the inverse do not hold since we consider in this
            // implementation that _other should always hold maximal `genericity`
            // due to `compatible_with` fn name sementics
            (TDim::Val(_dim), TDim::Sym(_other_dim)) => true,
            // for all other case equality is required
            (dim, other_dim) => dim == other_dim,
        }
    }
}

pub(super) fn reduce_ratio(mut p: i64, mut q: i64) -> (i64, u64) {
    use num_integer::Integer;
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
        Self::from(0)
    }
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

impl Default for TDim {
    fn default() -> TDim {
        TDim::zero()
    }
}

impl num_traits::Bounded for TDim {
    fn min_value() -> Self {
        TDim::Val(i64::min_value())
    }

    fn max_value() -> Self {
        TDim::Val(i64::max_value())
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
        TDim::Sym(*it)
    }
}

impl ops::Neg for TDim {
    type Output = Self;
    fn neg(self) -> Self {
        TDim::MulInt(-1, Box::new(self)).reduce()
    }
}

impl<'a> ops::AddAssign<&'a TDim> for TDim {
    fn add_assign(&mut self, rhs: &'a TDim) {
        *self = TDim::Add(vec![std::mem::take(self), rhs.clone()]).reduce()
    }
}

impl<I> ops::AddAssign<I> for TDim
where
    I: Into<TDim>,
{
    fn add_assign(&mut self, rhs: I) {
        let rhs = rhs.into();
        *self += &rhs
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

impl<'a> ops::SubAssign<&'a TDim> for TDim {
    fn sub_assign(&mut self, rhs: &'a TDim) {
        use std::ops::Neg;
        *self += rhs.clone().neg()
    }
}

impl<I> ops::SubAssign<I> for TDim
where
    I: Into<TDim>,
{
    fn sub_assign(&mut self, rhs: I) {
        use std::ops::Neg;
        *self += rhs.into().neg()
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
        *self = TDim::Mul(vec![rhs.into(), std::mem::take(self)]).reduce()
    }
}

impl<'a> ops::MulAssign<&'a TDim> for TDim {
    fn mul_assign(&mut self, rhs: &'a TDim) {
        *self = TDim::Mul(vec![std::mem::take(self), rhs.clone()]).reduce()
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

impl std::str::FromStr for TDim {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<TDim, Self::Err> {
        let first = s.chars().next().unwrap();
        if first.is_digit(10) || first == '-' {
            Ok(s.parse::<i64>()?.into())
        } else if first.is_alphabetic() && s.len() == 1 {
            Ok(Symbol::from(first).into())
        } else {
            anyhow::bail!("Can't parse {} as TDim", s)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! b( ($e:expr) => { Box::new($e) } );

    lazy_static::lazy_static! {
        static ref S: Symbol = crate::dim::Symbol::new('S');
    }

    fn s() -> TDim {
        (*S).into()
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
        assert_eq!(add(&s(), &neg(&s())).reduce(), Val(0))
    }

    #[test]
    fn reduce_neg_mul() {
        assert_eq!(neg(&mul(2, &s())).reduce(), mul(-2, &s()))
    }

    #[test]
    fn reduce_cplx_ex_2() {
        assert_eq!(
            add(&add(&Val(-4), &mul(-2, &div(&s(), 4))), &mul(-2, &mul(-1, &div(&s(), 4))))
                .reduce(),
            Val(-4)
        )
    }

    #[test]
    fn reduce_cplx_ex_3() {
        assert_eq!(div(&MulInt(1, b!(MulInt(4, b!(s())))), 4).reduce(), s())
    }

    #[test]
    fn reduce_cplx_ex_4() {
        // (S+1)/2 + (1-S)/2 == 1
        assert_eq!(
            add(&div(&add(&s(), &Val(1)), 2), &div(&add(&neg(&s()), &Val(1)), 2)).reduce(),
            1.into()
        );
    }

    #[test]
    fn reduce_mul_mul_1() {
        assert_eq!(mul(3, &mul(2, &s())).reduce(), mul(6, &s()))
    }

    #[test]
    fn reduce_mul_mul_2() {
        assert_eq!(mul(-2, &mul(-1, &s())).reduce(), mul(2, &s()))
    }

    #[test]
    fn reduce_mul_div_1() {
        assert_eq!(mul(2, &div(&mul(-1, &s()), 3)).reduce(), mul(-2, &div(&s(), 3)))
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
        let x = Symbol::new('x');
        let e: TDim = x.into();
        assert_eq!(e.eval(&SymbolValues::default().with(x, 2)).to_i64().unwrap(), 2);
        let e = e + 3;
        assert_eq!(e.eval(&SymbolValues::default().with(x, 2)).to_i64().unwrap(), 5);
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
        let e: TDim = Val(1) * s();
        assert_eq!(e, s());
        let e: TDim = s() * 'b' * 1;
        assert_eq!(e, s() * 'b');
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
        let e1: TDim = (s() + 23) / 2 - 1;
        let e2: TDim = (s() + 21) / 2;
        assert_eq!(e1, e2);
    }

    #[test]
    fn reduce_div_bug_1() {
        let e1: TDim = (s() + -1) / 2;
        let e2: TDim = (s() + 1) / 2 - 1;
        assert_eq!(e1, e2);
    }

    #[test]
    fn reduce_div_bug_2() {
        let e1: TDim = ((s() + 1) / 2 + 1) / 2;
        let e2: TDim = (s() + 3) / 4;
        assert_eq!(e1, e2);
    }

    #[test]
    fn reduce_div_bug_3() {
        let e1: TDim = (s() / 2) * -4;
        let e2: TDim = (s() / 2) * -4 / 1;
        assert_eq!(e1, e2);
    }

    #[test]
    fn reduce_mul_div() {
        let e: TDim = s() * 2 / 2;
        assert_eq!(e, s());
    }

    #[test]
    fn reduce_div_mul() {
        let e: TDim = s() / 2 * 2;
        assert_ne!(e, s());
    }

    #[test]
    fn reduce_add_div() {
        let e: TDim = s() / 2 + 1;
        assert_eq!(e, ((s() + 2) / 2));
    }

    #[test]
    fn reduce_neg_mul_() {
        let e: TDim = TDim::from(1) - s() * 2;
        assert_eq!(e, TDim::from(1) + s() * -2);
    }

    #[test]
    fn reduce_add_rem_1() {
        assert_eq!(((s() + 4) % 2), (s() % 2));
    }

    #[test]
    fn reduce_add_rem_2() {
        assert_eq!(((s() - 4) % 2), (s() % 2));
    }

    #[test]
    fn reduce_rem_div() {
        let e: TDim = s() % 2 / 2;
        assert_eq!(e, TDim::from(0));
    }

    #[test]
    fn conv2d_ex_1() {
        let e = (TDim::from(1) - 1 + 1).div_ceil(1);
        assert_eq!(e, TDim::from(1));
    }

    #[test]
    fn conv2d_ex_2() {
        let e = (s() - 3 + 1).div_ceil(1);
        assert_eq!(e, s() + -2);
    }
}
