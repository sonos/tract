use super::tree::ExpNode;
use crate::TractResult;
use std::{fmt, ops};

use std::collections::HashMap;
use crate::errors::TractResultExt;

use ExpNode::*;

#[derive(Clone, new, Hash)]
pub struct Stack {
    it: ExpNode,
}

impl fmt::Debug for Stack {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        self.it.fmt(fmt)
    }
}

impl fmt::Display for Stack {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        self.it.fmt(fmt)
    }
}

#[cfg(feature = "serialize")]
impl ::serde::Serialize for Stack {
    fn serialize<S>(&self, serializer: S) -> ::std::result::Result<S::Ok, S::Error>
    where
        S: ::serde::Serializer,
    {
        self.as_ops().serialize(serializer)
    }
}

impl Eq for Stack {}

impl PartialEq for Stack {
    fn eq(&self, other: &Stack) -> bool {
        self.it == other.it
    }
}

impl Stack {
    pub fn eval(&self, values: &HashMap<char, i32>) -> TractResult<i32> {
        self.it.eval(values).chain_err(|| format!("Evaluating {}", self.it))
    }

    pub fn sym(s: char) -> Stack {
        Stack::new(ExpNode::Sym(s))
    }

    pub fn div_ceil(mut self, rhs: u32) -> Stack {
        self.it = ExpNode::Div(Box::new(Add(vec![self.it, Val(rhs as i32 - 1)])), rhs).reduce();
        self
    }
}

impl From<i32> for Stack {
    fn from(v: i32) -> Stack {
        Stack::new(ExpNode::Val(v))
    }
}

impl From<char> for Stack {
    fn from(s: char) -> Stack {
        Stack::new(ExpNode::Sym(s))
    }
}

impl ops::Neg for Stack {
    type Output = Self;
    fn neg(mut self) -> Self {
        self.it = ExpNode::Mul(-1, Box::new(self.it)).reduce();
        self
    }
}

impl<'a> ops::AddAssign<&'a Stack> for Stack {
    fn add_assign(&mut self, rhs: &'a Stack) {
        let mut me = ExpNode::Val(0);
        std::mem::swap(&mut me, &mut self.it);
        self.it = ExpNode::Add(vec![me, rhs.it.clone()]).reduce();
    }
}

impl<I> ops::AddAssign<I> for Stack
where
    I: Into<Stack>,
{
    fn add_assign(&mut self, rhs: I) {
        let rhs = rhs.into();
        *self += &rhs
    }
}

impl<I> ops::Add<I> for Stack
where
    I: Into<Stack>,
{
    type Output = Self;
    fn add(mut self, rhs: I) -> Self {
        self += rhs;
        self
    }
}

impl<'a> ops::SubAssign<&'a Stack> for Stack {
    fn sub_assign(&mut self, rhs: &'a Stack) {
        use std::ops::Neg;
        *self += rhs.clone().neg()
    }
}

impl<I> ops::SubAssign<I> for Stack
where
    I: Into<Stack>,
{
    fn sub_assign(&mut self, rhs: I) {
        use std::ops::Neg;
        *self += rhs.into().neg()
    }
}

impl<I> ops::Sub<I> for Stack
where
    I: Into<Stack>,
{
    type Output = Self;
    fn sub(mut self, rhs: I) -> Self {
        self -= rhs;
        self
    }
}

impl ops::MulAssign<i32> for Stack {
    fn mul_assign(&mut self, rhs: i32) {
        let mut me = ExpNode::Val(0);
        std::mem::swap(&mut me, &mut self.it);
        self.it = ExpNode::Mul(rhs, Box::new(me)).reduce();
    }
}

impl ops::Mul<i32> for Stack {
    type Output = Self;
    fn mul(mut self, rhs: i32) -> Self {
        self *= rhs;
        self
    }
}

impl ops::DivAssign<u32> for Stack {
    fn div_assign(&mut self, rhs: u32) {
        let mut me = ExpNode::Val(0);
        std::mem::swap(&mut me, &mut self.it);
        self.it = ExpNode::Div(Box::new(me), rhs).reduce();
    }
}

impl ops::Div<u32> for Stack {
    type Output = Self;
    fn div(mut self, rhs: u32) -> Self {
        self /= rhs;
        self
    }
}

impl ops::RemAssign<u32> for Stack {
    fn rem_assign(&mut self, rhs: u32) {
        *self += -(self.clone() / rhs * rhs as i32);
    }
}

impl ops::Rem<u32> for Stack {
    type Output = Self;
    fn rem(mut self, rhs: u32) -> Self {
        self %= rhs;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn const_and_add() {
        let e: Stack = 2i32.into();
        assert_eq!(e.eval(&hashmap! {}).unwrap(), 2);
        let e: Stack = Stack::from(2) + 3;
        assert_eq!(e.eval(&hashmap! {}).unwrap(), 5);
        let e: Stack = Stack::from(2) - 3;
        assert_eq!(e.eval(&hashmap! {}).unwrap(), -1);
        let e: Stack = -Stack::from(2);
        assert_eq!(e.eval(&hashmap! {}).unwrap(), -2);
    }

    #[test]
    fn substitution() {
        let e = Stack::sym('x');
        assert_eq!(e.eval(&hashmap! {'x' => 2}).unwrap(), 2);
        let e = Stack::sym('x') + 3;
        assert_eq!(e.eval(&hashmap! {'x' => 2}).unwrap(), 5);
    }

    #[test]
    fn reduce_adds() {
        let e: Stack = Stack::from(2) + 1;
        assert_eq!(e, Stack::from(3));
        let e: Stack = Stack::from(3) + 2;
        assert_eq!(e, Stack::from(5));
        let e: Stack = Stack::from(3) + 0;
        assert_eq!(e, Stack::from(3));
        let e: Stack = Stack::from(3) + 2 + 1;
        assert_eq!(e, Stack::from(6));
    }

    #[test]
    fn reduce_divs() {
        let e: Stack = Stack::from(2) / 1;
        assert_eq!(e, Stack::from(2));
        let e: Stack = Stack::from(3) / 2;
        assert_eq!(e, Stack::from(1));
        let e: Stack = Stack::from(3) % 2;
        assert_eq!(e, Stack::from(1));
        let e: Stack = Stack::from(5) / 2;
        assert_eq!(e, Stack::from(2));
        let e: Stack = Stack::from(5) % 2;
        assert_eq!(e, Stack::from(1));
    }

    #[test]
    fn reduce_div_bug_0() {
        let e1: Stack = (Stack::sym('S') + 23) / 2 - 1;
        let e2: Stack = (Stack::sym('S') + 21) / 2;
        assert_eq!(e1, e2);
    }

    #[test]
    fn reduce_div_bug_1() {
        let e1: Stack = (Stack::sym('S') + -1) / 2;
        let e2: Stack = (Stack::sym('S') + 1) / 2 - 1;
        assert_eq!(e1, e2);
    }

    #[test]
    fn reduce_div_bug_2() {
        let e1: Stack = ((Stack::sym('S') + 1) / 2 + 1) / 2;
        let e2: Stack = (Stack::sym('S') + 3) / 4;
        assert_eq!(e1, e2);
    }

    #[test]
    fn reduce_div_bug_3() {
        let e1: Stack = (Stack::sym('S') / 2) * -4;
        let e2: Stack = (Stack::sym('S') / 2) * -4 / 1;
        assert_eq!(e1, e2);
    }

    #[test]
    fn reduce_mul_div() {
        let e: Stack = Stack::sym('S') * 2 / 2;
        assert_eq!(e, Stack::sym('S'));
    }

    #[test]
    fn reduce_div_mul() {
        let e: Stack = Stack::sym('S') / 2 * 2;
        assert_ne!(e, Stack::sym('S'));
    }

    #[test]
    fn reduce_add_div() {
        let e: Stack = Stack::sym('S') / 2 + 1;
        assert_eq!(e, ((Stack::sym('S') + 2) / 2));
    }

    #[test]
    fn reduce_neg_mul_() {
        let e: Stack = Stack::from(1) - Stack::sym('S') * 2;
        assert_eq!(e, Stack::from(1) + Stack::sym('S') * -2);
    }

    #[test]
    fn reduce_add_rem_1() {
        assert_eq!(((Stack::sym('S') + 4) % 2), (Stack::sym('S') % 2));
    }

    #[test]
    fn reduce_add_rem_2() {
        assert_eq!(((Stack::sym('S') - 4) % 2), (Stack::sym('S') % 2));
    }

    #[test]
    fn reduce_rem_div() {
        let e: Stack = Stack::sym('S') % 2 / 2;
        assert_eq!(e, Stack::from(0));
    }

    #[test]
    fn conv2d_ex_1() {
        let e = (Stack::from(1) - 1 + 1).div_ceil(1);
        assert_eq!(e, Stack::from(1));
    }

    #[test]
    fn conv2d_ex_2() {
        let e = (Stack::sym('S') - 3 + 1).div_ceil(1);
        assert_eq!(e, Stack::sym('S') + -2);
    }
}
