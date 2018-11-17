use pb::*;
use tract_core::*;

impl NodeProto {
    fn get_attr_opt(&self, name: &str) -> TractResult<Option<&AttributeProto>> {
        Ok(self.get_attribute().iter().find(|a| a.get_name() == name))
    }

    fn get_attr_opt_with_type(
        &self,
        name: &str,
        ty: AttributeProto_AttributeType,
    ) -> TractResult<Option<&AttributeProto>> {
        let attr = if let Some(a) = self.get_attr_opt(name)? {
            a
        } else {
            return Ok(None);
        };
        if attr.get_field_type() != ty {
            bail!(
                "Node {} ({}) expected attribute {} to be {:?}",
                self.get_name(),
                self.get_op_type(),
                name,
                ty
            )
        }
        Ok(Some(attr))
    }

    pub fn get_attr_opt_tensor(&self, name: &str) -> TractResult<Option<Tensor>> {
        match self.get_attr_opt_with_type(name, AttributeProto_AttributeType::TENSOR)? {
            Some(attr) => Ok(Some(attr.get_t().tractify()?)),
            None => Ok(None),
        }
    }

    pub fn get_attr_tensor(&self, name: &str) -> TractResult<Tensor> {
        Ok(self.get_attr_opt_tensor(name)?.ok_or_else(|| {
            format!(
                "Node {} ({}) expected tensor attribute '{}'",
                self.get_name(),
                self.get_op_type(),
                name
            )
        })?)
    }

    pub fn get_attr_opt_str(&self, name: &str) -> TractResult<Option<&str>> {
        match self.get_attr_opt_with_type(name, AttributeProto_AttributeType::STRING)? {
            Some(attr) => Ok(Some(::std::str::from_utf8(attr.get_s())?)),
            None => Ok(None),
        }
    }

    pub fn get_attr_str(&self, name: &str) -> TractResult<&str> {
        Ok(self.get_attr_opt_str(name)?.ok_or_else(|| {
            format!(
                "Node {} ({}) expected string attribute '{}'",
                self.get_name(),
                self.get_op_type(),
                name
            )
        })?)
    }

    pub fn get_attr_opt_int(&self, name: &str) -> TractResult<Option<i64>> {
        match self.get_attr_opt_with_type(name, AttributeProto_AttributeType::INT)? {
            Some(attr) => Ok(Some(attr.get_i())),
            None => Ok(None),
        }
    }

    pub fn get_attr_int(&self, name: &str) -> TractResult<i64> {
        Ok(self.get_attr_opt_int(name)?.ok_or_else(|| {
            format!(
                "Node {} ({}) expected int attribute '{}'",
                self.get_name(),
                self.get_op_type(),
                name
            )
        })?)
    }

    pub fn get_attr_opt_float(&self, name: &str) -> TractResult<Option<f32>> {
        match self.get_attr_opt_with_type(name, AttributeProto_AttributeType::FLOAT)? {
            Some(attr) => Ok(Some(attr.get_f())),
            None => Ok(None),
        }
    }

    pub fn get_attr_float(&self, name: &str) -> TractResult<f32> {
        Ok(self.get_attr_opt_float(name)?.ok_or_else(|| {
            format!(
                "Node {} ({}) expected float attribute '{}'",
                self.get_name(),
                self.get_op_type(),
                name
            )
        })?)
    }

    pub fn get_attr_opt_ints(&self, name: &str) -> TractResult<Option<&[i64]>> {
        match self.get_attr_opt_with_type(name, AttributeProto_AttributeType::INTS)? {
            Some(attr) => Ok(Some(attr.get_ints())),
            None => Ok(None),
        }
    }

    pub fn get_attr_ints(&self, name: &str) -> TractResult<&[i64]> {
        Ok(self.get_attr_opt_ints(name)?.ok_or_else(|| {
            format!(
                "Node {} ({}) expected list of ints attribute '{}'",
                self.get_name(),
                self.get_op_type(),
                name
            )
        })?)
    }
}
