use super::tree::ExpNode;
use crate::model::TVec;
use crate::TractResult;
use std::collections::HashMap;
use std::{fmt, ops};

#[derive(Clone)]
pub struct Stack(TVec<StackOp>);

#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
pub enum StackOp {
    Sym(char),
    Val(i32),
    Neg,
    Add,
    Div(u32),
    DivCeil(u32),
    Mul(i32),
    Rem(u32),
}

impl fmt::Debug for Stack {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self.format() {
            Ok(s) => write!(fmt, "{}", s),
            Err(e) => write!(fmt, "{:?}", e),
        }
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
        self.as_ops() == other.as_ops() || self.clone().reduce().as_ops() == other.clone().reduce().as_ops()
    }
}

impl Stack {
    pub fn empty() -> Stack {
        Stack(tvec!())
    }

    pub fn sym(s: char) -> Stack {
        Stack(tvec!(StackOp::Sym(s)))
    }

    pub fn eval(&self, values: &HashMap<char, i32>) -> TractResult<i32> {
        use self::StackOp::*;
        let mut stack = tvec![];
        for op in self.as_ops().iter() {
            match op {
                Val(v) => stack.push(*v),
                Sym(v) => stack.push(*values.get(v).ok_or(format!("Unresolved value {:?}", v))?),
                Neg => {
                    let a = stack.last_mut().ok_or("Too short stack")?;
                    *a = -*a;
                }
                Add => {
                    let b = stack.pop().ok_or("Too short stack")?;
                    *stack.last_mut().ok_or("Too short stack")? += b;
                }
                Mul(v) => {
                    *stack.last_mut().ok_or("Too short stack")? *= v;
                }
                Div(v) => {
                    *stack.last_mut().ok_or("Too short stack")? /= *v as i32;
                }
                DivCeil(v) => {
                    use num_integer::Integer;
                    let a = stack.pop().ok_or("Too short stack")?;
                    let (d, r) = a.div_rem(&(*v as i32));
                    stack.push(d + (r > 0) as i32);
                }
                Rem(v) => {
                    *stack.last_mut().ok_or("Too short stack")? %= *v as i32;
                }
            }
        }
        if stack.len() > 1 {
            bail!("Too long stack")
        }
        if stack.len() < 1 {
            bail!("Too short stack")
        }
        Ok(stack[0])
    }

    pub fn format(&self) -> TractResult<String> {
        Ok(format!("{}", ExpNode::from_ops(&self)))
    }

    pub fn as_ops(&self) -> &[StackOp] {
        &*self.0
    }

    pub fn push(&mut self, op: StackOp) {
        self.0.push(op)
    }

    pub fn push_all(&mut self, other: &[StackOp]) {
        for i in other {
            self.push(*i)
        }
    }

    pub fn val(&self) -> Option<&i32> {
        if let StackOp::Val(ref v) = &self.0[0] {
            if self.0.len() == 1 {
                return Some(v);
            }
        }
        return None;
    }

    pub fn mut_val(&mut self) -> Option<&mut i32> {
        if self.0.len() == 1 {
            if let StackOp::Val(ref mut v) = &mut self.0[0] {
                return Some(v);
            }
        }
        return None;
    }

    pub fn div_ceil(mut self, rhs: u32) -> Stack {
        self.0.push(StackOp::DivCeil(rhs));
        self.reduce()
    }

    pub fn to_tree(&self) -> ExpNode {
        ExpNode::from_ops(self)
    }

    fn reduce(self) -> Stack {
        self.to_tree().reduce().to_stack()
    }
}

impl From<i32> for Stack {
    fn from(v: i32) -> Stack {
        let mut e = Stack::empty();
        e.push(StackOp::Val(v));
        e
    }
}

impl From<char> for Stack {
    fn from(s: char) -> Stack {
        let mut e = Stack::empty();
        e.push(StackOp::Sym(s));
        e
    }
}

impl ops::Neg for Stack {
    type Output = Self;
    fn neg(mut self) -> Self {
        if let Some(v) = self.mut_val() {
            *v = -*v;
            return (*v).into();
        }
        self.push(StackOp::Neg);
        self.to_tree().reduce().to_stack()
    }
}

impl<'a> ops::AddAssign<&'a Stack> for Stack {
    fn add_assign(&mut self, rhs: &'a Stack) {
        if let (Some(lhs), Some(rhs)) = (self.mut_val(), rhs.val()) {
            *lhs += *rhs;
            return;
        }
        *self = ExpNode::Add(vec![self.to_tree(), rhs.to_tree()]).reduce().to_stack()
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
        *self = ExpNode::Mul(rhs, Box::new(self.to_tree())).reduce().to_stack()
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
        *self = ExpNode::Div(Box::new(self.to_tree()), rhs).reduce().to_stack()
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
        self.0.push(StackOp::Rem(rhs));
        self.0 = self.to_tree().reduce().to_stack().0
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
    use super::StackOp::*;
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
    fn simple_error_cases() {
        let e = Stack::empty();
        assert!(e.eval(&hashmap! {}).is_err());
        let mut e = Stack::empty();
        e.push(Val(2));
        e.push(Val(2));
        assert!(e.eval(&hashmap! {}).is_err());
        e.push(Val(2));
        e.push(Add);
        assert!(e.eval(&hashmap! {}).is_err());
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
        let e1:Stack = (Stack::sym('S') + 23) / 2 - 1;
        let e2:Stack = (Stack::sym('S') + 21) / 2;
        assert_eq!(e1, e2);
    }

    #[test]
    fn reduce_div_bug_1() {
        let e1:Stack = (Stack::sym('S') + -1) / 2;
        let e2:Stack = (Stack::sym('S') + 1) / 2 - 1;
        assert_eq!(e1.reduce(), e2.reduce());
    }

    #[test]
    fn reduce_div_bug_2() {
        let e1:Stack = ((Stack::sym('S') + 1) / 2 + 1) / 2;
        let e2:Stack = (Stack::sym('S') + 3) / 4;
        assert_eq!(e1, e2);
    }

    #[test]
    fn reduce_div_bug_3() {
        let e1:Stack = (Stack::sym('S') / 2) * -4 ;
        let e2:Stack = (Stack::sym('S') / 2) * -4 / 1;
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
        assert_eq!(e, ((Stack::sym('S') + 2) / 2).reduce());
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
