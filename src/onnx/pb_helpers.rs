use super::pb::*;
use insideout::InsideOut;
use Result;
use ToTfd;

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

    fn get_attr_opt_tensor(&self, name: &str) -> Result<Option<::tensor::Tensor>> {
        self.get_attr_opt_with_type(name, AttributeProto_AttributeType::TENSOR)?
            .map(|attr| attr.get_t().to_tfd())
            .inside_out()
    }

    pub fn get_attr_tensor(&self, name: &str) -> ::Result<::tensor::Tensor> {
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
