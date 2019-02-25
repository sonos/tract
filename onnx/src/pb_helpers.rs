use crate::pb::*;
use tract_core::*;

use std::borrow::Cow;

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

    pub fn expect_attr_ok_or_else<T, R: Reason>(
        &self, attr: &str, result: Option<T>, what: R,
    ) -> TractResult<T> {
        match result {
            Some(v) => Ok(v),
            None => Err(self.expect_attr(attr, false, what).unwrap_err()),
        }
    }

    fn get_attr_opt(&self, name: &str) -> TractResult<Option<&AttributeProto>> {
        Ok(self.get_attribute().iter().find(|a| a.get_name() == name))
    }

    fn get_attr_opt_with_type(
        &self, name: &str, ty: AttributeProto_AttributeType,
    ) -> TractResult<Option<&AttributeProto>> {
        let attr = if let Some(a) = self.get_attr_opt(name)? {
            a
        } else {
            return Ok(None);
        };
        self.expect_attr(name, attr.get_field_type() == ty, || format!("{:?}", ty))?;
        Ok(Some(attr))
    }

    pub fn get_attr_opt_tensor(&self, name: &str) -> TractResult<Option<Tensor>> {
        match self.get_attr_opt_with_type(name, AttributeProto_AttributeType::TENSOR)? {
            Some(attr) => Ok(Some(attr.get_t().tractify()?)),
            None => Ok(None),
        }
    }

    pub fn get_attr_tensor(&self, name: &str) -> TractResult<Tensor> {
        self.expect_attr_ok_or_else(name, self.get_attr_opt_tensor(name)?, "tensor")
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
