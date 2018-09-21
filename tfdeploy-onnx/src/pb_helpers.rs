use pb::*;
use tfdeploy::*;

impl NodeProto {
    fn get_attr_opt(&self, name: &str) -> Result<Option<&AttributeProto>> {
        Ok(self.get_attribute().iter().find(|a| a.get_name() == name))
    }

    fn get_attr_opt_with_type(
        &self,
        name: &str,
        ty: AttributeProto_AttributeType,
    ) -> Result<Option<&AttributeProto>> {
        let attr = if let Some(a) = self.get_attr_opt(name)? {
            a
        } else {
            return Ok(None);
        };
        if attr.get_field_type() != ty {
            bail!(
                "Node {} ({}) expected attribute to be {}",
                self.get_name(),
                self.get_op_type(),
                name
            )
        }
        Ok(Some(attr))
    }

    fn get_attr_opt_tensor(&self, name: &str) -> Result<Option<Tensor>> {
        match self.get_attr_opt_with_type(name, AttributeProto_AttributeType::TENSOR)? {
            Some(attr) => Ok(Some(attr.get_t().to_tfd()?)),
            None => Ok(None)
        }
    }

    pub fn get_attr_tensor(&self, name: &str) -> Result<Tensor> {
        Ok(self.get_attr_opt_tensor(name)?.ok_or_else(|| {
            format!(
                "Node {} ({}) expected tensor attribute '{}'",
                self.get_name(),
                self.get_op_type(),
                name
            )
        })?)
    }
}
