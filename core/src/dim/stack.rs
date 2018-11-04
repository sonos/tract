use super::tree::ExpNode;
use std::collections::HashMap;
use std::{fmt, ops};
use TractResult;

const EXP_LEN: usize = 16;

#[derive(Copy, Clone)]
pub struct Stack {
    array: [StackOp; EXP_LEN],
    len: usize,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
pub enum StackOp {
    Sym(char),
    Val(i32),
    Neg,
    Add,
    Div,
    DivCeil,
    Mul,
    Rem,
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
        self.as_ops() == other.as_ops()
    }
}

impl Stack {
    pub fn empty() -> Stack {
        Stack {
            len: 0,
            array: unsafe { ::std::mem::uninitialized() },
        }
    }

    pub fn sym(s: char) -> Stack {
        let mut e = Self::empty();
        e.push(StackOp::Sym(s));
        e
    }

    pub fn eval(&self, values: &HashMap<char, i32>) -> TractResult<i32> {
        use self::StackOp::*;
        let mut stack: Vec<i32> = vec![];
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
                Mul => {
                    let b = stack.pop().ok_or("Too short stack")?;
                    *stack.last_mut().ok_or("Too short stack")? *= b;
                }
                Div => {
                    let b = stack.pop().ok_or("Too short stack")?;
                    *stack.last_mut().ok_or("Too short stack")? /= b;
                }
                DivCeil => {
                    use num::Integer;
                    let b = stack.pop().ok_or("Too short stack")?;
                    let a = stack.pop().ok_or("Too short stack")?;
                    let (d, r) = a.div_rem(&b);
                    stack.push(d + (r > 0) as i32);
                }
                Rem => {
                    let b = stack.pop().ok_or("Too short stack")?;
                    *stack.last_mut().ok_or("Too short stack")? %= b;
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
        Ok(format!("{:?}", ExpNode::from_ops(&self)))
    }

    pub fn as_ops(&self) -> &[StackOp] {
        &self.array[0..self.len]
    }

    pub fn push(&mut self, op: StackOp) {
        if self.len == self.array.len() {
            panic!("Dimension expression stack overflow");
        }
        self.array[self.len] = op;
        self.len += 1;
    }

    pub fn push_all(&mut self, other: &[StackOp]) {
        for i in other {
            self.push(*i)
        }
    }

    pub fn val(&self) -> Option<&i32> {
        if let StackOp::Val(ref v) = &self.array[0] {
            if self.len == 1 {
                return Some(v);
            }
        }
        return None;
    }

    pub fn mut_val(&mut self) -> Option<&mut i32> {
        if let StackOp::Val(ref mut v) = &mut self.array[0] {
            if self.len == 1 {
                return Some(v);
            }
        }
        return None;
    }

    pub fn div_ceil(self, rhs: &Stack) -> Stack {
        ExpNode::DivCeil(Box::new(self.to_tree()), Box::new(rhs.to_tree()))
            .reduce()
            .to_stack()
    }

    pub fn to_tree(&self) -> ExpNode {
        ExpNode::from_ops(self)
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

impl<I> ops::AddAssign<I> for Stack
where
    I: Into<Stack>,
{
    fn add_assign(&mut self, rhs: I) {
        let rhs = rhs.into();
        if let (Some(lhs), Some(rhs)) = (self.mut_val(), rhs.val()) {
            *lhs += *rhs;
            return;
        }
        *self = ExpNode::Add(vec![self.to_tree(), rhs.to_tree()])
            .reduce()
            .to_stack()
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

impl<I> ops::SubAssign<I> for Stack
where
    I: Into<Stack>,
{
    fn sub_assign(&mut self, rhs: I) {
        use std::ops::Neg;
        *self = *self + rhs.into().neg()
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

impl<I> ops::MulAssign<I> for Stack
where
    I: Into<Stack>,
{
    fn mul_assign(&mut self, rhs: I) {
        *self = ExpNode::Mul(1, vec![self.to_tree(), rhs.into().to_tree()])
            .reduce()
            .to_stack()
    }
}

impl<I> ops::Mul<I> for Stack
where
    I: Into<Stack>,
{
    type Output = Self;
    fn mul(mut self, rhs: I) -> Self {
        self *= rhs;
        self
    }
}

impl<I> ops::DivAssign<I> for Stack
where
    I: Into<Stack>,
{
    fn div_assign(&mut self, rhs: I) {
        *self = ExpNode::Div(Box::new(self.to_tree()), Box::new(rhs.into().to_tree()))
            .reduce()
            .to_stack()
    }
}

impl<I> ops::Div<I> for Stack
where
    I: Into<Stack>,
{
    type Output = Self;
    fn div(mut self, rhs: I) -> Self {
        self /= rhs;
        self
    }
}

impl<I> ops::RemAssign<I> for Stack
where
    I: Into<Stack>,
{
    fn rem_assign(&mut self, rhs: I) {
        *self = ExpNode::Rem(Box::new(self.to_tree()), Box::new(rhs.into().to_tree()))
            .reduce()
            .to_stack()
    }
}

impl<I> ops::Rem<I> for Stack
where
    I: Into<Stack>,
{
    type Output = Self;
    fn rem(mut self, rhs: I) -> Self {
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
        assert_eq!(e.eval(&hashmap!{}).unwrap(), 2);
        let e: Stack = Stack::from(2) + 3;
        assert_eq!(e.eval(&hashmap!{}).unwrap(), 5);
        let e: Stack = Stack::from(2) - 3;
        assert_eq!(e.eval(&hashmap!{}).unwrap(), -1);
        let e: Stack = -Stack::from(2);
        assert_eq!(e.eval(&hashmap!{}).unwrap(), -2);
    }

    #[test]
    fn simple_error_cases() {
        let e = Stack::empty();
        assert!(e.eval(&hashmap!{}).is_err());
        let mut e = Stack::empty();
        e.push(Val(2));
        e.push(Val(2));
        assert!(e.eval(&hashmap!{}).is_err());
        e.push(Val(2));
        e.push(Add);
        assert!(e.eval(&hashmap!{}).is_err());
    }

    #[test]
    #[should_panic]
    fn overflow() {
        let mut e = Stack::from(2);
        for n in 1..EXP_LEN {
            e += Stack::sym('x') / n as i32;
        }
    }

    #[test]
    fn substitution() {
        let e = Stack::sym('x');
        assert_eq!(e.eval(&hashmap!{'x' => 2}).unwrap(), 2);
        let e = Stack::sym('x') + 3;
        assert_eq!(e.eval(&hashmap!{'x' => 2}).unwrap(), 5);
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
    fn reduce_mul_div() {
        let e: Stack = Stack::sym('S') * 2 / 2;
        assert_eq!(e, Stack::sym('S'));
    }

    #[test]
    fn reduce_div_mul() {
        let e: Stack = Stack::sym('S') / 2 * 2;
        assert_eq!(e, Stack::from(2) * (Stack::sym('S') / 2));
    }

    #[test]
    fn reduce_neg_mul_() {
        let e: Stack = Stack::from(1) - Stack::from(2) * Stack::sym('S');
        assert_eq!(e, Stack::from(1) + -Stack::from(2) * Stack::sym('S'));
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
        let e = (Stack::from(1) - 1 + 1).div_ceil(&1.into());
        assert_eq!(e, Stack::from(1));
    }

    #[test]
    fn conv2d_ex_2() {
        let e = (Stack::sym('S') - 3 + 1).div_ceil(&1.into());
        assert_eq!(e, Stack::sym('S') + -2);
    }

}
