use itertools::Itertools;
use std::fmt;

use super::stack::*;

macro_rules! b( ($e:expr) => { Box::new($e) } );

#[derive(Clone, PartialEq, Eq, Ord, PartialOrd, Hash, Debug)]
pub enum ExpNode {
    Sym(char),
    Val(i32),
    Add(Vec<ExpNode>),
    Mul(i32, Box<ExpNode>),
    Div(Box<ExpNode>, u32),
}

impl fmt::Display for ExpNode {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use self::ExpNode::*;
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
    pub fn from_ops(ops: &Stack) -> ExpNode {
        use ExpNode::*;
        let mut stack: Vec<ExpNode> = vec![];
        for op in ops.as_ops().iter() {
            match op {
                StackOp::Val(v) => stack.push(ExpNode::Val(*v)),
                StackOp::Sym(v) => stack.push(ExpNode::Sym(*v)),
                StackOp::Add => {
                    let b = stack.pop().expect("Too short stack");
                    let a = stack.pop().expect("Too short stack");
                    if let ExpNode::Add(mut items) = a {
                        items.push(b);
                        stack.push(ExpNode::Add(items));
                    } else {
                        stack.push(ExpNode::Add(vec![a, b]));
                    }
                }
                StackOp::Neg => {
                    let a = stack.pop().expect("Too short stack");
                    stack.push(Mul(-1, b![a]));
                }
                StackOp::Div(v) => {
                    let a = stack.pop().expect("Too short stack");
                    stack.push(Div(Box::new(a), *v));
                }
                StackOp::DivCeil(v) => {
                    let a = stack.pop().expect("Too short stack");
                    stack.push(Div(b!(Add(vec![a, Val(*v as i32), Val(-1)])), *v))
                }
                StackOp::Rem(v) => {
                    // a%b = a - b*(a/b)
                    let a = stack.pop().expect("Too short stack");
                    stack.push(Add(vec![a.clone(), Mul(-(*v as i32), b!(Div(b!(a), *v)))]))
                }
                StackOp::Mul(v) => {
                    let a = stack.pop().expect("Too short stack");
                    stack.push(ExpNode::Mul(*v, b![a]));
                }
            }
        }
        stack.remove(0)
    }

    pub fn to_stack(&self) -> Stack {
        match self {
            ExpNode::Val(i) => Stack::from(*i),
            ExpNode::Sym(c) => Stack::sym(*c),
            ExpNode::Add(vec) => {
                let (first, rest) = vec.split_first().expect("Empty add node");
                let mut it = first.to_stack();
                for other in rest {
                    it.push_all(other.to_stack().as_ops());
                    it.push(StackOp::Add);
                }
                it
            }
            ExpNode::Mul(v, inner) => {
                let mut it = inner.to_stack();
                it.push(StackOp::Mul(*v));
                it
            }
            ExpNode::Div(a, b) => {
                let mut it = a.to_stack();
                it.push(StackOp::Div(*b));
                it
            }
        }
    }

    pub fn reduce(self) -> ExpNode {
        self.simplify().wiggle().into_iter().map(|e| e.simplify()).min_by_key(|e| e.cost()).unwrap()
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
            Add(terms) => terms
                .iter()
                .map(|e| e.wiggle())
                .multi_cartesian_product()
                .map(|terms| Add(terms))
                .collect(),
            Mul(p, a) => {
                let mut forms = vec![];
                for a in a.wiggle() {
                    if let Add(a) = &a {
                        forms.push(Add(a.clone().into_iter().map(|a| Mul(*p, b!(a))).collect()))
                    }
                    forms.push(Mul(*p, b!(a)));
                }
                forms
            }
            Div(a, q) => {
                let mut forms = vec![];
                for a in a.wiggle() {
                    if let Add(a) = &a {
                        let (integer, non_integer): (Vec<_>, Vec<_>) =
                            a.clone().into_iter().partition(|a| a.gcd() % q == 0);
                        let mut terms = integer.iter().map(|i| i.div(*q)).collect::<Vec<_>>();
                        terms.push(Div(b!(Add(non_integer)), *q));
                        forms.push(Add(terms))
                    }
                    forms.push(Div(b!(a), *q));
                }
                forms
            }
        }
    }

    fn simplify(self) -> ExpNode {
        use self::ExpNode::*;
        use num_integer::Integer;
        match self {
            Add(mut terms) => {
                use std::collections::HashMap;
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
                let a = a.simplify();
                if p == 0 {
                    Val(0)
                } else if p == 1 {
                    a
                } else if let Val(p2) = a {
                    Val(p * p2)
                } else if let Mul(p2, a) = a {
                    Mul(p * p2, a)
                } else {
                    Mul(p, b!(a))
                }
            }
            Div(a, q) => {
                let a = a.simplify();
                if let Val(a) = a {
                    Val(a / q as i32)
                } else if let Mul(-1, a) = a {
                    Mul(-1, b!(Div(a, q)))
                } else {
                    let gcd = a.gcd().gcd(&q);
                    let a = a.div(gcd);
                    let q = q / gcd;
                    if q == 1 {
                        a
                    } else {
                        Div(b!(a), q)
                    }
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
}

#[cfg(test)]
mod tests {
    use super::ExpNode::*;
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
    fn back_and_forth_1() {
        let e = Stack::from(2) + 3;
        assert_eq!(e, ExpNode::from_ops(&e).to_stack());
    }

    #[test]
    fn back_and_forth_2() {
        let e = Stack::from(2) + 3;
        assert_eq!(e, ExpNode::from_ops(&e).to_stack());
    }

    #[test]
    fn back_and_forth_3() {
        let e = Stack::from(2) * 3;
        assert_eq!(e, ExpNode::from_ops(&e).to_stack());
    }

    #[test]
    fn back_and_forth_4() {
        let e = Stack::sym('S') * 3;
        assert_eq!(e, ExpNode::from_ops(&e).to_stack());
    }

    #[test]
    fn back_and_forth_5() {
        let e = Stack::from(5) / 2;
        assert_eq!(e, ExpNode::from_ops(&e).to_stack());
    }

    #[test]
    fn back_and_forth_6() {
        let e = Stack::from(5) % 2;
        assert_eq!(e, ExpNode::from_ops(&e).to_stack());
    }

    #[test]
    fn back_and_forth_7() {
        let e = Stack::from(5).div_ceil(2);
        assert_eq!(e, ExpNode::from_ops(&e).to_stack());
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
}
