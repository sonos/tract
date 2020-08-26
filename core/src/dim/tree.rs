use crate::prelude::TractResult;
use itertools::Itertools;
use num_traits::{AsPrimitive, Zero};
use std::collections::HashMap;
use std::sync::atomic::AtomicUsize;
use std::{fmt, ops};

macro_rules! b( ($e:expr) => { Box::new($e) } );

static SYMBOL_ID: AtomicUsize = AtomicUsize::new(1);

#[derive(Copy, Clone, PartialEq, Eq, Ord, PartialOrd, Hash, Debug)]
pub struct Symbol(char, usize);

impl Symbol {
    pub fn new(c: char) -> Symbol {
        Symbol(c, SYMBOL_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }
}

#[derive(Clone, PartialEq, Eq, Ord, PartialOrd, Hash, Debug)]
pub enum TDim {
    Sym(Symbol),
    Val(i64),
    Add(Vec<TDim>),
    Mul(i64, Box<TDim>),
    Div(Box<TDim>, u64),
}

use TDim::*;

impl fmt::Display for TDim {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match &self {
            Sym(sym) => write!(fmt, "{}", sym.0),
            Val(it) => write!(fmt, "{}", it),
            Add(it) => write!(fmt, "{}", it.iter().map(|x| format!("{}", x)).join("+")),
            Mul(a, b) => write!(fmt, "{}.{}", a, b),
            Div(a, b) => write!(fmt, "({})/{}", a, b),
        }
    }
}

impl TDim {
    pub fn is_one(&self) -> bool {
        self.as_const().map(|i| i == 1).unwrap_or(false)
    }

    /// The special value S, for streaming.
    pub fn s() -> TDim {
        TDim::Sym(Symbol('S', 0))
    }

    /// The special value S, for streaming.
    pub fn stream() -> TDim {
        Self::s()
    }

    /// Try to convert the value to an integer, if it does not contains S.
    pub fn as_const(&self) -> Option<i64> {
        self.to_integer().ok()
    }

    pub fn is_stream(&self) -> bool {
        self.to_integer().is_err()
    }

    pub fn to_integer(&self) -> TractResult<i64> {
        self.eval_with(&hashmap!())
    }

    pub fn eval(&self, s: i64) -> Option<i64> {
        self.eval_with(&hashmap!(Symbol('S',0) => s)).ok()
    }

    fn eval_with(&self, values: &HashMap<Symbol, i64>) -> TractResult<i64> {
        Ok(match self {
            Sym(sym) => *values.get(&sym).ok_or(format!("Unresolved value {:?}", sym))?,
            Val(v) => *v,
            Add(terms) => terms.iter().try_fold(0i64, |acc, it| -> TractResult<i64> {
                Ok(acc + it.eval_with(values)?)
            })?,
            Div(a, q) => a.eval_with(values)? / *q as i64,
            Mul(p, a) => p * a.eval_with(values)?,
        })
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
            Div(a, _) => 3 * a.cost(),
            Mul(_, a) => 2 * a.cost(),
        }
    }

    fn wiggle(&self) -> Vec<TDim> {
        use self::TDim::*;
        match self {
            Sym(_) | Val(_) => vec![self.clone()],
            Add(terms) => {
                let mut forms = vec![];
                let sub_wiggle = terms.iter().map(|e| e.wiggle()).multi_cartesian_product();
                for sub in sub_wiggle {
                    for (ix, num, q) in sub
                        .iter()
                        .enumerate()
                        .filter_map(
                            |(ix, t)| if let Div(a, q) = t { Some((ix, a, q)) } else { None },
                        )
                        .next()
                    {
                        let new_num = sub
                            .iter()
                            .enumerate()
                            .map(|(ix2, t)| {
                                if ix2 != ix {
                                    Mul(*q as i64, b!(t.clone()))
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
            Mul(p, a) => a.wiggle().into_iter().map(|a| Mul(*p, b!(a))).collect(),
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
                        Mul(v, f) => {
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
                            Some(Mul(v, b![k]))
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
            Mul(p, a) => {
                if let Mul(p2, a) = *a {
                    return Mul(p * p2, a).simplify();
                } else if let Val(p2) = *a {
                    return Val(p * p2);
                }
                let a = a.simplify();
                if p == 0 {
                    Val(0)
                } else if p == 1 {
                    a
                } else if let Add(terms) = &a {
                    Add(terms.clone().into_iter().map(|a| Mul(p, b!(a)).simplify()).collect())
                } else if let Val(p2) = a {
                    Val(p * p2)
                } else if let Mul(p2, a) = a {
                    Mul(p * p2, a)
                } else {
                    Mul(p, b!(a))
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
                } else if let Mul(-1, a) = a {
                    Mul(-1, b!(Div(a, q)))
                } else if let Add(mut terms) = a {
                    if terms.iter().any(|t| t == &Mul(-1, b!(Self::s()))) {
                        Mul(
                            -1,
                            b!(Div(
                                b!(Add(terms.into_iter().map(|t| Mul(-1, b!(t))).collect())
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
                            Some(-(-v).div_ceil(&(q as i64)))
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
                } else if let Mul(p, a) = a {
                    if p == q as i64 {
                        a.simplify()
                    } else {
                        let gcd = p.abs().gcd(&(q as i64));
                        if gcd == p {
                            Div(a, q / gcd as u64)
                        } else if gcd == q as i64 {
                            Mul(p / gcd, a)
                        } else if gcd > 1 {
                            Div(b!(Mul(p / gcd, a)), q / gcd as u64).simplify()
                        } else {
                            Div(b!(Mul(p, a)), q)
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
            Mul(p, a) => a.gcd() * p.abs() as u64,
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
            Mul(p, a) => {
                if *p == d as i64 {
                    (**a).clone()
                } else {
                    let gcd = (p.abs() as u64).gcd(&d);
                    Mul(p / gcd as i64, b!(a.div(d / gcd)))
                }
            }
            Div(a, q) => Div(a.clone(), q * d),
        }
    }

    pub fn div_ceil(self, rhs: u64) -> TDim {
        TDim::Div(Box::new(Add(vec![self, Val(rhs as i64 - 1)])), rhs).reduce()
    }

    pub fn slope(&self) -> (i64, u64) {
        fn slope_rec(d: &TDim) -> (i64, i64) {
            match d {
                Val(_) => (0, 1),
                Sym(_) => (1, 1),
                Add(terms) => terms
                    .iter()
                    .map(slope_rec)
                    .fold((1, 1), |a, b| ((a.0 * b.1 + a.1 * b.0), (b.1 * a.1))),
                Mul(p, a) => {
                    let (n, d) = slope_rec(a);
                    (p * n, d)
                }
                Div(a, q) => {
                    let (n, d) = slope_rec(a);
                    (n, d * *q as i64)
                }
            }
        }
        let (p, q) = slope_rec(self);
        reduce_ratio(p, q)
    }
}

pub(super) fn reduce_ratio(mut p: i64, mut q: i64) -> (i64, u64) {
    use crate::num_integer::Integer;
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

impl ::std::iter::Sum for TDim {
    fn sum<I: Iterator<Item = TDim>>(iter: I) -> TDim {
        iter.fold(0.into(), |a, b| a + b)
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
        TDim::Mul(-1, Box::new(self)).reduce()
    }
}

impl<'a> ops::AddAssign<&'a TDim> for TDim {
    fn add_assign(&mut self, rhs: &'a TDim) {
        let mut swap = TDim::Val(0);
        std::mem::swap(&mut swap, self);
        *self = TDim::Add(vec![swap, rhs.clone()]).reduce()
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

impl ops::MulAssign<i64> for TDim {
    fn mul_assign(&mut self, rhs: i64) {
        let mut me = TDim::Val(0);
        std::mem::swap(&mut me, self);
        *self = TDim::Mul(rhs, Box::new(me)).reduce()
    }
}

impl<I: AsPrimitive<i64>> ops::Mul<I> for TDim {
    type Output = Self;
    fn mul(mut self, rhs: I) -> Self {
        self *= rhs.as_();
        self
    }
}

impl<I: AsPrimitive<u64>> ops::DivAssign<I> for TDim {
    fn div_assign(&mut self, rhs: I) {
        let mut me = TDim::Val(0);
        std::mem::swap(&mut me, self);
        *self = TDim::Div(Box::new(me), rhs.as_()).reduce()
    }
}

impl<I: AsPrimitive<u64>> ops::Div<I> for TDim {
    type Output = Self;
    fn div(mut self, rhs: I) -> Self {
        self /= rhs.as_();
        self
    }
}

impl<I: AsPrimitive<u64>> ops::RemAssign<I> for TDim {
    fn rem_assign(&mut self, rhs: I) {
        *self += -(self.clone() / rhs.as_() * rhs.as_());
    }
}

impl<I: AsPrimitive<u64>> ops::Rem<I> for TDim {
    type Output = Self;
    fn rem(mut self, rhs: I) -> Self {
        self %= rhs;
        self
    }
}

impl std::str::FromStr for TDim {
    type Err = std::num::ParseIntError;
    fn from_str(s: &str) -> Result<TDim, Self::Err> {
        if s == "S" {
            Ok(TDim::s())
        } else if s.ends_with("S") {
            let number: String = s.chars().take_while(|c| c.is_digit(10)).collect();
            let number: i64 = number.parse::<i64>().map(|i| i.into())?;
            Ok(TDim::s() * number)
        } else {
            s.parse::<i64>().map(|i| i.into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! b( ($e:expr) => { Box::new($e) } );

    fn s() -> TDim {
        TDim::Sym(Symbol('S', 0))
    }

    fn neg(a: &TDim) -> TDim {
        mul(-1, a)
    }

    fn add(a: &TDim, b: &TDim) -> TDim {
        TDim::Add(vec![a.clone(), b.clone()])
    }

    fn mul(a: i64, b: &TDim) -> TDim {
        TDim::Mul(a, b![b.clone()])
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
        assert_eq!(div(&Mul(1, b!(Mul(4, b!(s())))), 4).reduce(), s())
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
        assert_eq!(e.eval_with(&hashmap! {}).unwrap(), 2);
        let e: TDim = TDim::from(2) + 3;
        assert_eq!(e.eval_with(&hashmap! {}).unwrap(), 5);
        let e: TDim = TDim::from(2) - 3;
        assert_eq!(e.eval_with(&hashmap! {}).unwrap(), -1);
        let e: TDim = -TDim::from(2);
        assert_eq!(e.eval_with(&hashmap! {}).unwrap(), -2);
    }

    #[test]
    fn substitution() {
        let x = Symbol::new('x');
        let e:TDim = x.into();
        assert_eq!(e.eval_with(&hashmap! {x => 2}).unwrap(), 2);
        let e = e + 3;
        assert_eq!(e.eval_with(&hashmap! {x => 2}).unwrap(), 5);
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
