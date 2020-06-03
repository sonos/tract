use crate::prelude::TractResult;
use itertools::Itertools;
use std::collections::HashMap;
use std::{fmt, ops};

macro_rules! b( ($e:expr) => { Box::new($e) } );

#[derive(Clone, PartialEq, Eq, Ord, PartialOrd, Hash, Debug)]
pub enum ExpNode {
    Sym(char),
    Val(i32),
    Add(Vec<ExpNode>),
    Mul(i32, Box<ExpNode>),
    Div(Box<ExpNode>, u32),
}

use ExpNode::*;

impl fmt::Display for ExpNode {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match &self {
            Sym(it) => write!(fmt, "{}", it),
            Val(it) => write!(fmt, "{}", it),
            Add(it) => write!(fmt, "{}", it.iter().map(|x| format!("{}", x)).join("+")),
            Mul(a, b) => write!(fmt, "{}.{}", a, b),
            Div(a, b) => write!(fmt, "({})/{}", a, b),
        }
    }
}

impl ExpNode {
    pub fn eval(&self, values: &HashMap<char, i32>) -> TractResult<i32> {
        Ok(match self {
            Sym(v) => *values.get(v).ok_or(format!("Unresolved value {:?}", v))?,
            Val(v) => *v,
            Add(terms) => terms
                .iter()
                .try_fold(0i32, |acc, it| -> TractResult<i32> { Ok(acc + it.eval(values)?) })?,
            Div(a, q) => a.eval(values)? / *q as i32,
            Mul(p, a) => p * a.eval(values)?,
        })
    }

    pub fn reduce(self) -> ExpNode {
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
        use self::ExpNode::*;
        match self {
            Sym(_) | Val(_) => 1,
            Add(terms) => 2 * terms.iter().map(ExpNode::cost).sum::<usize>(),
            Div(a, _) => 3 * a.cost(),
            Mul(_, a) => 2 * a.cost(),
        }
    }

    fn wiggle(&self) -> Vec<ExpNode> {
        use self::ExpNode::*;
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
                                    Mul(*q as i32, b!(t.clone()))
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

    pub fn simplify(self) -> ExpNode {
        use self::ExpNode::*;
        use num_integer::Integer;
        match self {
            Add(mut terms) => {
                let mut reduced: HashMap<ExpNode, i32> = HashMap::new();
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
                    Val(a / q as i32)
                } else if let Mul(-1, a) = a {
                    Mul(-1, b!(Div(a, q)))
                } else if let Add(mut terms) = a {
                    if let Some(v) = terms
                        .iter()
                        .filter_map(|t| if let Val(v) = t { Some(*v) } else { None })
                        .next()
                    {
                        let offset = if v >= q as i32 {
                            Some(v / q as i32)
                        } else if v < 0 {
                            Some(-(-v).div_ceil(&(q as i32)))
                        } else {
                            None
                        };
                        if let Some(val) = offset {
                            terms.push(Val(-val * q as i32));
                            Add(vec![Val(val), Div(b!(Add(terms).simplify()), q)])
                        } else {
                            Div(b!(Add(terms)), q)
                        }
                    } else {
                        Div(b!(Add(terms)), q)
                    }
                } else if let Mul(p, a) = a {
                    if p == q as i32 {
                        a.simplify()
                    } else {
                        let gcd = p.abs().gcd(&(q as i32));
                        if gcd == p {
                            Div(a, q / gcd as u32)
                        } else if gcd == q as i32 {
                            Mul(p / gcd, a)
                        } else if gcd > 1 {
                            Div(b!(Mul(p / gcd, a)), q / gcd as u32).simplify()
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

    fn gcd(&self) -> u32 {
        use self::ExpNode::*;
        use num_integer::Integer;
        match self {
            Val(v) => v.abs() as u32,
            Sym(_) => 1,
            Add(terms) => {
                let (head, tail) = terms.split_first().unwrap();
                tail.iter().fold(head.gcd(), |a, b| a.gcd(&b.gcd()))
            }
            Mul(p, a) => a.gcd() * p.abs() as u32,
            Div(a, q) => {
                if a.gcd() % *q == 0 {
                    a.gcd() / *q
                } else {
                    1
                }
            }
        }
    }

    fn div(&self, d: u32) -> ExpNode {
        use self::ExpNode::*;
        use num_integer::Integer;
        if d == 1 {
            return self.clone();
        }
        match self {
            Val(v) => Val(v / d as i32),
            Sym(_) => panic!(),
            Add(terms) => Add(terms.iter().map(|t| t.div(d)).collect()),
            Mul(p, a) => {
                if *p == d as i32 {
                    (**a).clone()
                } else {
                    let gcd = (p.abs() as u32).gcd(&d);
                    Mul(p / gcd as i32, b!(a.div(d / gcd)))
                }
            }
            Div(a, q) => Div(a.clone(), q * d),
        }
    }

    pub fn div_ceil(self, rhs: u32) -> ExpNode {
        ExpNode::Div(Box::new(Add(vec![self, Val(rhs as i32 - 1)])), rhs).reduce()
    }

}

impl From<i32> for ExpNode {
    fn from(v: i32) -> ExpNode {
        ExpNode::Val(v)
    }
}

impl ops::Neg for ExpNode {
    type Output = Self;
    fn neg(self) -> Self {
        ExpNode::Mul(-1, Box::new(self)).reduce()
    }
}

impl<'a> ops::AddAssign<&'a ExpNode> for ExpNode {
    fn add_assign(&mut self, rhs: &'a ExpNode) {
        let mut swap = ExpNode::Val(0);
        std::mem::swap(&mut swap, self);
        *self = ExpNode::Add(vec![swap, rhs.clone()]).reduce()
    }
}

impl<I> ops::AddAssign<I> for ExpNode
where
    I: Into<ExpNode>,
{
    fn add_assign(&mut self, rhs: I) {
        let rhs = rhs.into();
        *self += &rhs
    }
}

impl<I> ops::Add<I> for ExpNode
where
    I: Into<ExpNode>,
{
    type Output = Self;
    fn add(mut self, rhs: I) -> Self {
        self += rhs;
        self
    }
}

impl<'a> ops::SubAssign<&'a ExpNode> for ExpNode {
    fn sub_assign(&mut self, rhs: &'a ExpNode) {
        use std::ops::Neg;
        *self += rhs.clone().neg()
    }
}

impl<I> ops::SubAssign<I> for ExpNode
where
    I: Into<ExpNode>,
{
    fn sub_assign(&mut self, rhs: I) {
        use std::ops::Neg;
        *self += rhs.into().neg()
    }
}

impl<I> ops::Sub<I> for ExpNode
where
    I: Into<ExpNode>,
{
    type Output = Self;
    fn sub(mut self, rhs: I) -> Self {
        self -= rhs;
        self
    }
}

impl ops::MulAssign<i32> for ExpNode {
    fn mul_assign(&mut self, rhs: i32) {
        let mut me = ExpNode::Val(0);
        std::mem::swap(&mut me, self);
        *self = ExpNode::Mul(rhs, Box::new(me)).reduce()
    }
}

impl ops::Mul<i32> for ExpNode {
    type Output = Self;
    fn mul(mut self, rhs: i32) -> Self {
        self *= rhs;
        self
    }
}

impl ops::DivAssign<u32> for ExpNode {
    fn div_assign(&mut self, rhs: u32) {
        let mut me = ExpNode::Val(0);
        std::mem::swap(&mut me, self);
        *self = ExpNode::Div(Box::new(me), rhs).reduce()
    }
}

impl ops::Div<u32> for ExpNode {
    type Output = Self;
    fn div(mut self, rhs: u32) -> Self {
        self /= rhs;
        self
    }
}

impl ops::RemAssign<u32> for ExpNode {
    fn rem_assign(&mut self, rhs: u32) {
        *self += -(self.clone() / rhs * rhs as i32);
    }
}

impl ops::Rem<u32> for ExpNode {
    type Output = Self;
    fn rem(mut self, rhs: u32) -> Self {
        self %= rhs;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! b( ($e:expr) => { Box::new($e) } );

    fn neg(a: &ExpNode) -> ExpNode {
        mul(-1, a)
    }

    fn add(a: &ExpNode, b: &ExpNode) -> ExpNode {
        ExpNode::Add(vec![a.clone(), b.clone()])
    }

    fn mul(a: i32, b: &ExpNode) -> ExpNode {
        ExpNode::Mul(a, b![b.clone()])
    }

    fn div(a: &ExpNode, b: u32) -> ExpNode {
        ExpNode::Div(b!(a.clone()), b)
    }

    #[test]
    fn reduce_add() {
        assert_eq!(add(&Sym('S'), &neg(&Sym('S'))).reduce(), Val(0))
    }

    #[test]
    fn reduce_neg_mul() {
        assert_eq!(neg(&mul(2, &Sym('S'))).reduce(), mul(-2, &Sym('S')))
    }

    #[test]
    fn reduce_cplx_ex_2() {
        assert_eq!(
            add(
                &add(&Val(-4), &mul(-2, &div(&Sym('S'), 4))),
                &mul(-2, &mul(-1, &div(&Sym('S'), 4)))
            )
            .reduce(),
            Val(-4)
        )
    }

    #[test]
    fn reduce_cplx_ex_3() {
        assert_eq!(div(&Mul(1, b!(Mul(4, b!(Sym('S'))))), 4).reduce(), Sym('S'))
    }

    #[test]
    fn reduce_mul_mul_1() {
        assert_eq!(mul(3, &mul(2, &Sym('S'))).reduce(), mul(6, &Sym('S')))
    }

    #[test]
    fn reduce_mul_mul_2() {
        assert_eq!(mul(-2, &mul(-1, &Sym('S'))).reduce(), mul(2, &Sym('S')))
    }

    #[test]
    fn reduce_mul_div_1() {
        assert_eq!(mul(2, &div(&mul(-1, &Sym('S')), 3)).reduce(), mul(-2, &div(&Sym('S'), 3)))
    }

    #[test]
    fn const_and_add() {
        let e: ExpNode = 2i32.into();
        assert_eq!(e.eval(&hashmap! {}).unwrap(), 2);
        let e: ExpNode = ExpNode::from(2) + 3;
        assert_eq!(e.eval(&hashmap! {}).unwrap(), 5);
        let e: ExpNode = ExpNode::from(2) - 3;
        assert_eq!(e.eval(&hashmap! {}).unwrap(), -1);
        let e: ExpNode = -ExpNode::from(2);
        assert_eq!(e.eval(&hashmap! {}).unwrap(), -2);
    }

    #[test]
    fn substitution() {
        let e = ExpNode::Sym('x');
        assert_eq!(e.eval(&hashmap! {'x' => 2}).unwrap(), 2);
        let e = ExpNode::Sym('x') + 3;
        assert_eq!(e.eval(&hashmap! {'x' => 2}).unwrap(), 5);
    }

    #[test]
    fn reduce_adds() {
        let e: ExpNode = ExpNode::from(2) + 1;
        assert_eq!(e, ExpNode::from(3));
        let e: ExpNode = ExpNode::from(3) + 2;
        assert_eq!(e, ExpNode::from(5));
        let e: ExpNode = ExpNode::from(3) + 0;
        assert_eq!(e, ExpNode::from(3));
        let e: ExpNode = ExpNode::from(3) + 2 + 1;
        assert_eq!(e, ExpNode::from(6));
    }

    #[test]
    fn reduce_divs() {
        let e: ExpNode = ExpNode::from(2) / 1;
        assert_eq!(e, ExpNode::from(2));
        let e: ExpNode = ExpNode::from(3) / 2;
        assert_eq!(e, ExpNode::from(1));
        let e: ExpNode = ExpNode::from(3) % 2;
        assert_eq!(e, ExpNode::from(1));
        let e: ExpNode = ExpNode::from(5) / 2;
        assert_eq!(e, ExpNode::from(2));
        let e: ExpNode = ExpNode::from(5) % 2;
        assert_eq!(e, ExpNode::from(1));
    }

    #[test]
    fn reduce_div_bug_0() {
        let e1: ExpNode = (ExpNode::Sym('S') + 23) / 2 - 1;
        let e2: ExpNode = (ExpNode::Sym('S') + 21) / 2;
        assert_eq!(e1, e2);
    }

    #[test]
    fn reduce_div_bug_1() {
        let e1: ExpNode = (ExpNode::Sym('S') + -1) / 2;
        let e2: ExpNode = (ExpNode::Sym('S') + 1) / 2 - 1;
        assert_eq!(e1, e2);
    }

    #[test]
    fn reduce_div_bug_2() {
        let e1: ExpNode = ((ExpNode::Sym('S') + 1) / 2 + 1) / 2;
        let e2: ExpNode = (ExpNode::Sym('S') + 3) / 4;
        assert_eq!(e1, e2);
    }

    #[test]
    fn reduce_div_bug_3() {
        let e1: ExpNode = (ExpNode::Sym('S') / 2) * -4;
        let e2: ExpNode = (ExpNode::Sym('S') / 2) * -4 / 1;
        assert_eq!(e1, e2);
    }

    #[test]
    fn reduce_mul_div() {
        let e: ExpNode = ExpNode::Sym('S') * 2 / 2;
        assert_eq!(e, ExpNode::Sym('S'));
    }

    #[test]
    fn reduce_div_mul() {
        let e: ExpNode = ExpNode::Sym('S') / 2 * 2;
        assert_ne!(e, ExpNode::Sym('S'));
    }

    #[test]
    fn reduce_add_div() {
        let e: ExpNode = ExpNode::Sym('S') / 2 + 1;
        assert_eq!(e, ((ExpNode::Sym('S') + 2) / 2));
    }

    #[test]
    fn reduce_neg_mul_() {
        let e: ExpNode = ExpNode::from(1) - ExpNode::Sym('S') * 2;
        assert_eq!(e, ExpNode::from(1) + ExpNode::Sym('S') * -2);
    }

    #[test]
    fn reduce_add_rem_1() {
        assert_eq!(((ExpNode::Sym('S') + 4) % 2), (ExpNode::Sym('S') % 2));
    }

    #[test]
    fn reduce_add_rem_2() {
        assert_eq!(((ExpNode::Sym('S') - 4) % 2), (ExpNode::Sym('S') % 2));
    }

    #[test]
    fn reduce_rem_div() {
        let e: ExpNode = ExpNode::Sym('S') % 2 / 2;
        assert_eq!(e, ExpNode::from(0));
    }

    #[test]
    fn conv2d_ex_1() {
        let e = (ExpNode::from(1) - 1 + 1).div_ceil(1);
        assert_eq!(e, ExpNode::from(1));
    }

    #[test]
    fn conv2d_ex_2() {
        let e = (ExpNode::Sym('S') - 3 + 1).div_ceil(1);
        assert_eq!(e, ExpNode::Sym('S') + -2);
    }
}
