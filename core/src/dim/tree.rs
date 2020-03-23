use itertools::Itertools;
use std::fmt;

use super::stack::*;

macro_rules! b( ($e:expr) => { Box::new($e) } );

#[derive(Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub enum ExpNode {
    Sym(char),
    Val(i32),
    Add(Vec<ExpNode>),
    Mul(i32, Vec<ExpNode>),
    Div(Box<ExpNode>, u32),
    Rem(Box<ExpNode>, u32),
    DivCeil(Box<ExpNode>, u32),
}

impl fmt::Debug for ExpNode {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use self::ExpNode::*;
        match &self {
            Sym(it) => write!(fmt, "{}", it),
            Val(it) => write!(fmt, "{}", it),
            Add(it) => write!(fmt, "({})", it.iter().map(|x| format!("{:?}", x)).join("+")),
            Mul(a, b) if *a == 1 => {
                write!(fmt, "{}", b.iter().map(|x| format!("{:?}", x)).join("*"))
            }
            Mul(a, b) => write!(fmt, "{}.{}", a, b.iter().map(|x| format!("{:?}", x)).join("")),
            Div(a, b) => write!(fmt, "{:?}/{:?}", a, b),
            Rem(a, b) => write!(fmt, "{:?}%{:?}", a, b),
            DivCeil(a, b) => write!(fmt, "({:?}/{:?}).ceil()", a, b),
        }
    }
}

impl ExpNode {
    pub fn from_ops(ops: &Stack) -> ExpNode {
        use self::StackOp::*;
        let mut stack: Vec<ExpNode> = vec![];
        for op in ops.as_ops().iter() {
            match op {
                Val(v) => stack.push(ExpNode::Val(*v)),
                Sym(v) => stack.push(ExpNode::Sym(*v)),
                Add => {
                    let b = stack.pop().expect("Too short stack");
                    let a = stack.pop().expect("Too short stack");
                    if let ExpNode::Add(mut items) = a {
                        items.push(b);
                        stack.push(ExpNode::Add(items));
                    } else {
                        stack.push(ExpNode::Add(vec![a, b]));
                    }
                }
                Neg => {
                    let a = stack.pop().expect("Too short stack");
                    stack.push(ExpNode::Mul(-1, vec![a]));
                }
                Div(v) => {
                    let a = stack.pop().expect("Too short stack");
                    stack.push(ExpNode::Div(Box::new(a), *v));
                }
                DivCeil(v) => {
                    let a = stack.pop().expect("Too short stack");
                    stack.push(ExpNode::DivCeil(Box::new(a), *v));
                }
                Rem(v) => {
                    let a = stack.pop().expect("Too short stack");
                    stack.push(ExpNode::Rem(b!(a), *v));
                }
                Mul(v) => {
                    let a = stack.pop().expect("Too short stack");
                    stack.push(ExpNode::Mul(*v, vec![a]));
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
            ExpNode::Mul(v, vec) => {
                assert!(vec.len() == 1);
                let mut it = vec[0].to_stack();
                it.push(StackOp::Mul(*v));
                it
                /*
                if *v != 1 {
                    it.push(StackOp::Val(*v));
                }
                let (first, rest) = vec.split_first().expect("Empty mul node");
                it.push_all(first.to_stack().as_ops());
                for other in rest {
                    it.push_all(other.to_stack().as_ops());
                    it.push(StackOp::Mul);
                }
                if *v != 1 {
                    it.push(StackOp::Mul);
                }
                it
                */
            }
            ExpNode::Div(a, b) => {
                let mut it = a.to_stack();
                it.push(StackOp::Div(*b));
                it
            }
            ExpNode::Rem(a, b) => {
                let mut it = a.to_stack();
                it.push(StackOp::Rem(*b));
                it
            }
            ExpNode::DivCeil(a, b) => {
                let mut it = a.to_stack();
                it.push(StackOp::DivCeil(*b));
                it
            }
        }
    }

    pub fn reduce(self) -> ExpNode {
        use self::ExpNode::*;
        let res = match self {
            Div(a, b) => {
                let red_a = a.reduce();
                match (red_a, b) {
                    (a, 1) => a,
                    (Val(a), b) => Val(a / b as i32),
                    (Add(vals), b) => {
                        let mut out: Vec<ExpNode> = vec![];
                        let mut kept: Vec<ExpNode> = vec![];
                        for val in vals {
                            match val {
                                Val(num) if num % b as i32 == 0 => {
                                    out.push(Val(num / b as i32));
                                    continue;
                                }
                                Mul(m, factors) => {
                                    if m % b as i32 == 0 {
                                        out.push(Mul(m / b as i32, factors));
                                    } else {
                                        kept.push(Mul(m, factors))
                                    }
                                }
                                v => kept.push(v),
                            }
                        }
                        if kept.len() == 1 {
                            out.push(Div(b!(kept.remove(0)), b));
                        } else if kept.len() > 1 {
                            kept.sort();
                            out.push(Div(b!(Add(kept)), b));
                        }
                        if out.len() == 1 {
                            out.remove(0)
                        } else {
                            out.sort();
                            Add(out).reduce()
                        }
                    }
                    (Mul(v, factors), b) => {
                        use num_integer::Integer;
                        let gcd = v.abs().gcd(&(b as i32));
                        if gcd == b as i32 {
                            Mul(v / gcd, factors).reduce()
                        } else if gcd == 1 {
                            Mul(v, vec![Div(b!(Mul(1, factors).reduce()), b)])
                        } else {
                            Div(b!(Mul(v / gcd, factors)), b / gcd as u32)
                        }
                    }
                    (a, b) => Div(b!(a), b),
                }
            }
            Rem(a, b) => {
                // a%b = a - b*(a/b)
                let a = a.reduce();
                if b == 1 {
                    ExpNode::Val(0)
                } else {
                    Add(vec![a.clone(), Mul(-(b as i32), vec![Div(b!(a.clone()), b)])]).reduce()
                }
            }
            DivCeil(a, b) => {
                // ceiling(j/m) = floor(j+m-1/m)
                let red_a = a.reduce();
                Div(b!(Add(vec![red_a, Val(b as i32), Val(-1)])), b).reduce()
            }
            Add(mut vec) => {
                use std::collections::HashMap;
                let mut reduced: HashMap<ExpNode, i32> = HashMap::new();
                while let Some(item) = vec.pop() {
                    let red = item.reduce();
                    match red {
                        Add(items) => {
                            vec.extend(items.into_iter());
                            continue;
                        }
                        Val(0) => (),
                        Val(v) => *reduced.entry(Val(1)).or_insert(0) += v,
                        Mul(v, mut f) => {
                            if f.len() == 1 {
                                *reduced.entry(f.remove(0)).or_insert(0) += v;
                            } else {
                                *reduced.entry(Mul(1, f)).or_insert(0) += v;
                            }
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
                            Some(Mul(v, vec![k]))
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
            Mul(scale, mut vec) => {
                let mut reduced = vec![];
                let mut value = scale;
                vec.reverse();
                while let Some(item) = vec.pop() {
                    let red: ExpNode = item.reduce();
                    if let Val(v) = red {
                        value *= v;
                    } else if let Mul(v, items) = red {
                        value *= v;
                        vec.extend(items.into_iter());
                    } else {
                        reduced.push(red);
                    }
                }
                if value == 0 {
                    Val(0)
                } else if reduced.len() == 0 {
                    Val(value)
                } else if reduced.len() == 1 {
                    let item = reduced.remove(0);
                    if let Add(items) = item {
                        let mut items = items
                            .into_iter()
                            .map(|f| Mul(value, vec![f]).reduce())
                            .collect::<Vec<ExpNode>>();
                        items.sort();
                        Add(items)
                    } else {
                        if value == 1 {
                            item
                        } else {
                            Mul(value, vec![item])
                        }
                    }
                } else {
                    Mul(value, reduced)
                }
            }
            it => it,
        };

        res
    }
}

#[cfg(test)]
mod tests {
    use super::ExpNode::*;
    use super::*;

    fn neg(a: &ExpNode) -> ExpNode {
        mul(-1, a)
    }

    fn rem(a: &ExpNode, b: u32) -> ExpNode {
        ExpNode::Rem(Box::new(a.clone()), b)
    }

    fn add(a: &ExpNode, b: &ExpNode) -> ExpNode {
        ExpNode::Add(vec![a.clone(), b.clone()])
    }

    fn mul(a: i32, b: &ExpNode) -> ExpNode {
        ExpNode::Mul(a, vec![b.clone()])
    }

    fn div(a: &ExpNode, b: u32) -> ExpNode {
        ExpNode::Div(Box::new(a.clone()), b)
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
    fn reduce_cplx_ex_1() {
        assert_eq!(
            Add(vec![
                add(&Sym('S'), &Val(-4)),
                add(&Val(4), &Mul(1, vec![Val(-2), div(&Sym('S'), 2)])),
            ])
            .reduce(),
            add(&Sym('S'), &mul(-2, &div(&Sym('S'), 2)))
        )
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
        assert_eq!(div(&Mul(1, vec![Sym('S'), Val(4)]), 4).reduce(), Sym('S'))
    }

    #[test]
    fn reduce_cplx_ex_4() {
        assert_eq!(
            div(
                &Mul(1, vec![add(&Val(-4), &Mul(1, vec![Val(-8), div(&Sym('S'), 8)])), Val(8),]),
                8
            )
            .reduce(),
            add(&Val(-4), &mul(-8, &div(&Sym('S'), 8)))
        )
    }

    #[test]
    fn reduce_cplx_ex_5() {
        assert_eq!(
            mul(-1, &add(&Sym('S'), &Val(-182))).reduce(),
            add(&Mul(1, vec![Val(-1), Sym('S')]), &Val(182)).reduce(),
        )
    }

    #[test]
    fn reduce_mul_1() {
        assert_eq!(Mul(1, vec![Val(2), Sym('S')]).reduce(), Mul(2, vec![Sym('S')]));
        assert_eq!(Mul(1, vec![Sym('S'), Val(2)]).reduce(), Mul(2, vec![Sym('S')]));
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
    fn reduce_rem_div() {
        assert_eq!(div(&rem(&Sym('S'), 2), 2).reduce(), Val(0))
    }
}
