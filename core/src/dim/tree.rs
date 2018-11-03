use itertools::Itertools;
use std::fmt;

use super::stack::*;

#[derive(Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub enum ExpNode {
    Sym(char),
    Val(i32),
    Add(Vec<ExpNode>),
    Mul(i32, Vec<ExpNode>),
    Div(Box<ExpNode>, Box<ExpNode>),
    Rem(Box<ExpNode>, Box<ExpNode>),
    DivCeil(Box<ExpNode>, Box<ExpNode>),
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
            Mul(a, b) => write!(
                fmt,
                "{}.{}",
                a,
                b.iter().map(|x| format!("{:?}", x)).join("")
            ),
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
                Div => {
                    let b = stack.pop().expect("Too short stack");
                    let a = stack.pop().expect("Too short stack");
                    stack.push(ExpNode::Div(Box::new(a), Box::new(b)));
                }
                DivCeil => {
                    let b = stack.pop().expect("Too short stack");
                    let a = stack.pop().expect("Too short stack");
                    stack.push(ExpNode::DivCeil(Box::new(a), Box::new(b)));
                }
                Rem => {
                    let b = stack.pop().expect("Too short stack");
                    let a = stack.pop().expect("Too short stack");
                    stack.push(ExpNode::Rem(Box::new(a), Box::new(b)));
                }
                Mul => {
                    let b = stack.pop().expect("Too short stack");
                    let a = stack.pop().expect("Too short stack");
                    stack.push(ExpNode::Mul(1, vec![a, b]));
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
                let mut it = Stack::empty();
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
            }
            ExpNode::Div(a, b) => {
                let mut it = a.to_stack();
                it.push_all(b.to_stack().as_ops());
                it.push(StackOp::Div);
                it
            }
            ExpNode::Rem(a, b) => {
                let mut it = a.to_stack();
                it.push_all(b.to_stack().as_ops());
                it.push(StackOp::Rem);
                it
            }
            ExpNode::DivCeil(a, b) => {
                let mut it = a.to_stack();
                it.push_all(b.to_stack().as_ops());
                it.push(StackOp::DivCeil);
                it
            }
        }
    }

    pub fn reduce(self) -> ExpNode {
        macro_rules! b( ($e:expr) => { Box::new($e) } );
        use self::ExpNode::*;
        let res = match self {
            Div(a, b) => {
                let red_a = a.reduce();
                let red_b = b.reduce();
                match (red_a, red_b) {
                    (a, Val(1)) => a,
                    (Val(a), Val(b)) => Val(a / b),
                    (Add(vals), Val(b)) => {
                        let mut out: Vec<ExpNode> = vec![];
                        let mut kept: Vec<ExpNode> = vec![];
                        for val in vals {
                            match val {
                                Val(num) if num % b == 0 => {
                                    out.push(Val(num / b));
                                    continue;
                                }
                                Mul(m, factors) => {
                                    if m % b == 0 {
                                        out.push(Mul(m / b, factors));
                                    } else {
                                        kept.push(Mul(m, factors))
                                    }
                                }
                                v => kept.push(v),
                            }
                        }
                        if kept.len() == 1 {
                            out.push(Div(b!(kept.remove(0)), b!(Val(b))));
                        } else if kept.len() > 1 {
                            kept.sort();
                            out.push(Div(b!(Add(kept)), b!(Val(b))));
                        }
                        if out.len() == 1 {
                            out.remove(0)
                        } else {
                            out.sort();
                            Add(out).reduce()
                        }
                    }
                    (Mul(v, factors), Val(b)) => {
                        use num::Integer;
                        let gcd = v.gcd(&b);
                        if gcd == b {
                            Mul(v / gcd, factors).reduce()
                        } else if gcd == 1 {
                            Mul(v, vec![Div(b!(Mul(1, factors).reduce()), b!(Val(b)))])
                        } else {
                            Div(b!(Mul(v / gcd, factors)), b!(Val(b / gcd)))
                        }
                    }
                    (a, b) => Div(b!(a), b!(b)),
                }
            }
            Rem(a, b) => {
                // a%b = a - b*(a/b)
                let a = a.reduce();
                let b = b.reduce();
                if b == ExpNode::Val(1) {
                    a
                } else {
                    Add(vec![
                        a.clone(),
                        Mul(-1, vec![b.clone(), Div(b!(a.clone()), b!(b))]),
                    ]).reduce()
                }
            }
            DivCeil(a, b) => {
                // ceiling(j/m) = floor(j+m-1/m)
                let red_a = a.reduce();
                let red_b = b.reduce();
                Div(b!(Add(vec![red_a, red_b.clone(), Val(-1)])), b!(red_b)).reduce()
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
                    }).collect();
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

    fn rem(a: &ExpNode, b: &ExpNode) -> ExpNode {
        ExpNode::Rem(Box::new(a.clone()), Box::new(b.clone()))
    }

    fn add(a: &ExpNode, b: &ExpNode) -> ExpNode {
        ExpNode::Add(vec![a.clone(), b.clone()])
    }

    fn mul(a: i32, b: &ExpNode) -> ExpNode {
        ExpNode::Mul(a, vec![b.clone()])
    }

    fn div(a: &ExpNode, b: &ExpNode) -> ExpNode {
        ExpNode::Div(Box::new(a.clone()), Box::new(b.clone()))
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
        let e = Stack::from(5).div_ceil(&2.into());
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
                add(&Val(4), &Mul(1, vec![Val(-2), div(&Sym('S'), &Val(2))])),
            ]).reduce(),
            add(&Sym('S'), &mul(-2, &div(&Sym('S'), &Val(2))))
        )
    }

    #[test]
    fn reduce_cplx_ex_2() {
        assert_eq!(
            add(
                &add(&Val(-4), &mul(-2, &div(&Sym('S'), &Val(4)))),
                &mul(-2, &mul(-1, &div(&Sym('S'), &Val(4))))
            ).reduce(),
            Val(-4)
        )
    }

    #[test]
    fn reduce_cplx_ex_3() {
        assert_eq!(
            div(&Mul(1, vec![Sym('S'), Val(4)]), &Val(4)).reduce(),
            Sym('S')
        )
    }

    #[test]
    fn reduce_cplx_ex_4() {
        assert_eq!(
            div(
                &Mul(
                    1,
                    vec![
                        add(&Val(-4), &Mul(1, vec![Val(-8), div(&Sym('S'), &Val(8))])),
                        Val(8),
                    ]
                ),
                &Val(8)
            ).reduce(),
            add(&Val(-4), &mul(-8, &div(&Sym('S'), &Val(8))))
        )
    }

    #[test]
    fn reduce_cplx_ex_5() {
        assert_eq!(
            mul(-1, &add(&Sym('S'), &Val(-182))).reduce(),
            add(&Mul(1, vec![Val(-1), Sym('S')]), &Val(182))
                .reduce(),
        )
    }

    #[test]
    fn reduce_mul_1() {
        assert_eq!(
            Mul(1, vec![Val(2), Sym('S')]).reduce(),
            Mul(2, vec![Sym('S')])
        );
        assert_eq!(
            Mul(1, vec![Sym('S'), Val(2)]).reduce(),
            Mul(2, vec![Sym('S')])
        );
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
        assert_eq!(
            mul(2, &div(&mul(-1, &Sym('S')), &Val(3))).reduce(),
            mul(-2, &div(&Sym('S'), &Val(3)))
        )
    }

    #[test]
    fn reduce_rem_div() {
        assert_eq!(div(&rem(&Sym('S'), &Val(2)), &Val(2)).reduce(), Val(0))
    }
}
