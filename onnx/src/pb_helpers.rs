use crate::pb::*;
use tract_core::*;

use std::borrow::Cow;
use std::fmt::{self, Display};
use std::str;

pub trait TryCollect<T, E>: Iterator<Item = Result<T, E>> + Sized {
    #[must_use]
    fn try_collect<B: Default + Extend<T>>(mut self) -> Result<B, E> {
        let mut out = B::default();
        while let Some(item) = self.next() {
            out.extend(Some(item?));
        }
        Ok(out)
    }
}

impl<T, E, I> TryCollect<T, E> for I where I: Iterator<Item = Result<T, E>> + Sized {}

pub trait Reason {
    fn reason(&self) -> Cow<str>;
}

impl<'a> Reason for &'a str {
    fn reason(&self) -> Cow<str> {
        (*self).into()
    }
}

impl<F> Reason for F
where
    F: Fn() -> String,
{
    fn reason(&self) -> Cow<str> {
        self().into()
    }
}

trait OptionExt {
    type Item;

    fn and_try<F, T>(self, f: F) -> TractResult<Option<T>>
    where
        F: Fn(Self::Item) -> TractResult<T>;

    fn and_ok<F, T>(self, f: F) -> TractResult<Option<T>>
    where
        F: Fn(Self::Item) -> T;
}

impl<A> OptionExt for Option<A> {
    type Item = A;

    fn and_try<F, T>(self, f: F) -> TractResult<Option<T>>
    where
        F: Fn(Self::Item) -> TractResult<T>,
    {
        match self {
            Some(attr) => f(attr).map(Some),
            None => Ok(None),
        }
    }

    fn and_ok<F, T>(self, f: F) -> TractResult<Option<T>>
    where
        F: Fn(Self::Item) -> T,
    {
        Ok(self.map(f))
    }
}

impl Display for AttributeProto_AttributeType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            AttributeProto_AttributeType::INT => "int",
            AttributeProto_AttributeType::FLOAT => "float",
            AttributeProto_AttributeType::TENSOR => "tensor",
            AttributeProto_AttributeType::STRING => "string",
            AttributeProto_AttributeType::INTS => "list of ints",
            AttributeProto_AttributeType::FLOATS => "list of floats",
            AttributeProto_AttributeType::TENSORS => "list of tensors",
            AttributeProto_AttributeType::STRINGS => "list of strings",
            AttributeProto_AttributeType::GRAPH => "graph",
            AttributeProto_AttributeType::GRAPHS => "graphs",
            _ => "<undefined>",
        })
    }
}

pub trait AttrScalarType<'a>: 'a + Sized {
    fn get_attr_opt_scalar(node: &'a NodeProto, name: &str) -> TractResult<Option<Self>>;
}

impl<'a> AttrScalarType<'a> for Tensor {
    fn get_attr_opt_scalar(node: &'a NodeProto, name: &str) -> TractResult<Option<Self>> {
        node.get_attr_opt_with_type(name, AttributeProto_AttributeType::TENSOR)?
            .and_try(|attr| attr.get_t().tractify())
    }
}

impl<'a> AttrScalarType<'a> for &'a [u8] {
    fn get_attr_opt_scalar(node: &'a NodeProto, name: &str) -> TractResult<Option<Self>> {
        node.get_attr_opt_with_type(name, AttributeProto_AttributeType::STRING)?
            .and_ok(AttributeProto::get_s)
    }
}

impl<'a> AttrScalarType<'a> for &'a str {
    fn get_attr_opt_scalar(node: &'a NodeProto, name: &str) -> TractResult<Option<Self>> {
        let bytes: Option<&[u8]> = AttrScalarType::get_attr_opt_scalar(node, name)?;
        bytes.and_try(|b| str::from_utf8(b).map_err(Into::into))
    }
}

impl<'a> AttrScalarType<'a> for String {
    fn get_attr_opt_scalar(node: &'a NodeProto, name: &str) -> TractResult<Option<Self>> {
        let string: Option<&'a str> = AttrScalarType::get_attr_opt_scalar(node, name)?;
        string.and_ok(Into::into)
    }
}

impl<'a> AttrScalarType<'a> for i64 {
    fn get_attr_opt_scalar(node: &'a NodeProto, name: &str) -> TractResult<Option<Self>> {
        node.get_attr_opt_with_type(name, AttributeProto_AttributeType::INT)?
            .and_ok(AttributeProto::get_i)
    }
}

impl<'a> AttrScalarType<'a> for usize {
    fn get_attr_opt_scalar(node: &'a NodeProto, name: &str) -> TractResult<Option<Self>> {
        let int: Option<i64> = AttrScalarType::get_attr_opt_scalar(node, name)?;
        int.and_try(|int| {
            node.expect_attr(name, int >= 0, "non-negative int")?;
            Ok(int as _)
        })
    }
}

impl<'a> AttrScalarType<'a> for f32 {
    fn get_attr_opt_scalar(node: &'a NodeProto, name: &str) -> TractResult<Option<Self>> {
        node.get_attr_opt_with_type(name, AttributeProto_AttributeType::FLOAT)?
            .and_ok(AttributeProto::get_f)
    }
}

pub trait AttrSliceType<'a>: 'a + Sized {
    fn get_attr_opt_slice(node: &'a NodeProto, name: &str) -> TractResult<Option<&'a [Self]>>;
}

impl<'a> AttrSliceType<'a> for Vec<u8> {
    fn get_attr_opt_slice(node: &'a NodeProto, name: &str) -> TractResult<Option<&'a [Self]>> {
        node.get_attr_opt_with_type(name, AttributeProto_AttributeType::STRINGS)?
            .and_ok(AttributeProto::get_strings)
    }
}

impl<'a> AttrSliceType<'a> for i64 {
    fn get_attr_opt_slice(node: &'a NodeProto, name: &str) -> TractResult<Option<&'a [Self]>> {
        node.get_attr_opt_with_type(name, AttributeProto_AttributeType::INTS)?
            .and_ok(AttributeProto::get_ints)
    }
}

impl<'a> AttrSliceType<'a> for f32 {
    fn get_attr_opt_slice(node: &'a NodeProto, name: &str) -> TractResult<Option<&'a [Self]>> {
        node.get_attr_opt_with_type(name, AttributeProto_AttributeType::FLOATS)?
            .and_ok(AttributeProto::get_floats)
    }
}

pub trait AttrTVecType<'a>: 'a + Sized {
    fn get_attr_opt_tvec(node: &'a NodeProto, name: &str) -> TractResult<Option<TVec<Self>>>;
}

impl<'a, T> AttrTVecType<'a> for T
where
    T: AttrSliceType<'a> + Clone,
{
    fn get_attr_opt_tvec(node: &'a NodeProto, name: &str) -> TractResult<Option<TVec<Self>>> {
        T::get_attr_opt_slice(node, name)?.and_ok(Into::into)
    }
}

impl<'a> AttrTVecType<'a> for Tensor {
    fn get_attr_opt_tvec(node: &'a NodeProto, name: &str) -> TractResult<Option<TVec<Self>>> {
        node.get_attr_opt_with_type(name, AttributeProto_AttributeType::TENSORS)?
            .and_try(|attr| attr.get_tensors().iter().map(|t| t.tractify()).try_collect())
    }
}

impl<'a> AttrTVecType<'a> for &'a str {
    fn get_attr_opt_tvec(node: &'a NodeProto, name: &str) -> TractResult<Option<TVec<Self>>> {
        <Vec<u8>>::get_attr_opt_slice(node, name)?
            .and_try(|b| b.iter().map(|v| str::from_utf8(v)).try_collect().map_err(Into::into))
    }
}

impl<'a> AttrTVecType<'a> for String {
    fn get_attr_opt_tvec(node: &'a NodeProto, name: &str) -> TractResult<Option<TVec<Self>>> {
        <Vec<u8>>::get_attr_opt_slice(node, name)?.and_try(|b| {
            b.iter().map(|v| str::from_utf8(v).map(Into::into)).try_collect().map_err(Into::into)
        })
    }
}

impl<'a> AttrTVecType<'a> for usize {
    fn get_attr_opt_tvec(node: &'a NodeProto, name: &str) -> TractResult<Option<TVec<Self>>> {
        let ints: Option<&[i64]> = AttrSliceType::get_attr_opt_slice(node, name)?;
        ints.and_try(|ints| {
            for int in ints.iter() {
                node.expect_attr(name, *int >= 0, "list of non-negative ints")?;
            }
            Ok(ints.iter().map(|&x| x as _).collect())
        })
    }
}

impl NodeProto {
    pub fn expect<R: Reason>(&self, cond: bool, what: R) -> TractResult<()> {
        ensure!(
            cond,
            "Node {} ({}): expected {}",
            self.get_name(),
            self.get_op_type(),
            what.reason()
        );
        Ok(())
    }

    pub fn expect_attr<R: Reason>(&self, attr: &str, cond: bool, what: R) -> TractResult<()> {
        ensure!(
            cond,
            "Node {} ({}), attribute '{}': expected {}",
            self.get_name(),
            self.get_op_type(),
            attr,
            what.reason()
        );
        Ok(())
    }

    pub fn expect_ok_or_else<T, R: Reason>(&self, result: Option<T>, what: R) -> TractResult<T> {
        match result {
            Some(v) => Ok(v),
            None => Err(self.expect(false, what).unwrap_err()),
        }
    }

    pub fn expect_attr_ok_or_else<T, R: Reason>(
        &self, attr: &str, result: Option<T>, what: R,
    ) -> TractResult<T> {
        match result {
            Some(v) => Ok(v),
            None => Err(self.expect_attr(attr, false, what).unwrap_err()),
        }
    }

    fn get_attr_opt_with_type(
        &self, name: &str, ty: AttributeProto_AttributeType,
    ) -> TractResult<Option<&AttributeProto>> {
        let attr = match self.get_attribute().iter().find(|a| a.get_name() == name) {
            Some(attr) => attr,
            _ => return Ok(None),
        };
        self.expect_attr(name, attr.get_field_type() == ty, || {
            format!("{}, got {}", ty, attr.get_field_type())
        })?;
        Ok(Some(attr))
    }

    pub fn get_attr_opt<'a, T>(&'a self, name: &str) -> TractResult<Option<T>>
    where
        T: AttrScalarType<'a>,
    {
        T::get_attr_opt_scalar(self, name)
    }

    pub fn get_attr<'a, T>(&'a self, name: &str) -> TractResult<T>
    where
        T: AttrScalarType<'a>,
    {
        self.expect_ok_or_else(self.get_attr_opt(name)?, || format!("attribute '{}'", name))
    }

    pub fn get_attr_opt_slice<'a, T>(&'a self, name: &str) -> TractResult<Option<&'a [T]>>
    where
        T: AttrSliceType<'a>,
    {
        T::get_attr_opt_slice(self, name)
    }

    pub fn get_attr_slice<'a, T>(&'a self, name: &str) -> TractResult<&'a [T]>
    where
        T: AttrSliceType<'a>,
    {
        self.expect_ok_or_else(self.get_attr_opt_slice(name)?, || format!("attribute '{}'", name))
    }

    pub fn get_attr_opt_tvec<'a, T>(&'a self, name: &str) -> TractResult<Option<TVec<T>>>
    where
        T: AttrTVecType<'a>,
    {
        T::get_attr_opt_tvec(self, name)
    }

    pub fn get_attr_tvec<'a, T>(&'a self, name: &str) -> TractResult<TVec<T>>
    where
        T: AttrTVecType<'a>,
    {
        self.expect_ok_or_else(self.get_attr_opt_tvec(name)?, || format!("attribute '{}'", name))
    }

    pub fn get_attr_opt_str(&self, name: &str) -> TractResult<Option<&str>> {
        match self.get_attr_opt_with_type(name, AttributeProto_AttributeType::STRING)? {
            Some(attr) => Ok(Some(::std::str::from_utf8(attr.get_s())?)),
            None => Ok(None),
        }
    }

    pub fn get_attr_str(&self, name: &str) -> TractResult<&str> {
        self.expect_attr_ok_or_else(name, self.get_attr_opt_str(name)?, "string")
    }

    pub fn get_attr_opt_int(&self, name: &str) -> TractResult<Option<i64>> {
        match self.get_attr_opt_with_type(name, AttributeProto_AttributeType::INT)? {
            Some(attr) => Ok(Some(attr.get_i())),
            None => Ok(None),
        }
    }

    pub fn get_attr_int(&self, name: &str) -> TractResult<i64> {
        self.expect_attr_ok_or_else(name, self.get_attr_opt_int(name)?, "int")
    }

    pub fn get_attr_opt_float(&self, name: &str) -> TractResult<Option<f32>> {
        match self.get_attr_opt_with_type(name, AttributeProto_AttributeType::FLOAT)? {
            Some(attr) => Ok(Some(attr.get_f())),
            None => Ok(None),
        }
    }

    pub fn get_attr_float(&self, name: &str) -> TractResult<f32> {
        self.expect_attr_ok_or_else(name, self.get_attr_opt_float(name)?, "float")
    }

    pub fn get_attr_opt_ints(&self, name: &str) -> TractResult<Option<&[i64]>> {
        match self.get_attr_opt_with_type(name, AttributeProto_AttributeType::INTS)? {
            Some(attr) => Ok(Some(attr.get_ints())),
            None => Ok(None),
        }
    }

    pub fn get_attr_ints(&self, name: &str) -> TractResult<&[i64]> {
        self.expect_attr_ok_or_else(name, self.get_attr_opt_ints(name)?, "list of ints")
    }

    pub fn get_attr_opt_floats(&self, name: &str) -> TractResult<Option<&[f32]>> {
        match self.get_attr_opt_with_type(name, AttributeProto_AttributeType::FLOATS)? {
            Some(attr) => Ok(Some(attr.get_floats())),
            None => Ok(None),
        }
    }

    pub fn get_attr_floats(&self, name: &str) -> TractResult<&[f32]> {
        self.expect_attr_ok_or_else(name, self.get_attr_opt_floats(name)?, "list of floats")
    }

    pub fn get_attr_usize_tvec(&self, name: &str) -> TractResult<TVec<usize>> {
        let ints = self.get_attr_ints(name)?;
        for i in ints.iter() {
            self.expect_attr(name, *i >= 0, "list of non-negative ints")?;
        }
        Ok(ints.iter().map(|&x| x as _).collect())
    }

    pub fn get_attr_opt_int_tvec(&self, name: &str) -> TractResult<Option<TVec<i64>>> {
        Ok(self.get_attr_opt_ints(name)?.map(Into::into))
    }

    pub fn get_attr_int_tvec(&self, name: &str) -> TractResult<TVec<i64>> {
        Ok(self.get_attr_ints(name)?.into())
    }

    pub fn get_attr_opt_float_tvec(&self, name: &str) -> TractResult<Option<TVec<f32>>> {
        Ok(self.get_attr_opt_floats(name)?.map(Into::into))
    }

    pub fn get_attr_float_tvec(&self, name: &str) -> TractResult<TVec<f32>> {
        Ok(self.get_attr_floats(name)?.into())
    }
}
