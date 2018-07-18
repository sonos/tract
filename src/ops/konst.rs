use std::collections::HashMap;

use super::{Attr, Op, OpRegister, TensorView};
use analyser::TensorFact;
use std::sync::Arc;
use tfpb::types::DataType;
use {Result, Tensor};

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("Const", Const::build);
}

#[derive(Debug, Clone)]
pub struct Const {
    dtype: DataType,
    value: Arc<Tensor>,
}

impl Const {
    pub fn build(node: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        let dtype = node.get_attr_datatype("dtype")?;
        let mat = node.get_attr_tensor("value")?;

        if mat.datatype() != dtype {
            bail!(
                "Const node {:?} doesn't have the expected {:?} type.",
                mat,
                dtype
            );
        }

        Ok(Box::new(Const {
            dtype,
            value: Arc::new(mat),
        }))
    }
}

impl Op for Const {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, _inputs: Vec<TensorView>) -> Result<Vec<TensorView>> {
        Ok(vec![self.value.clone().into()])
    }

    /// Returns the attributes of the operation and their values.
    fn get_attributes(&self) -> HashMap<&'static str, Attr> {
        hashmap!{
            "dtype" => Attr::DataType(self.dtype),
            "value" => Attr::Tensor(self.value.as_ref().clone()),
        }
    }

    /// Infers properties about the input and output tensors.
    fn infer(
        &self,
        _: Vec<TensorFact>,
        _: Vec<TensorFact>,
    ) -> Result<(Vec<TensorFact>, Vec<TensorFact>)> {
        let output = TensorFact {
            datatype: typefact!(self.dtype),
            shape: self.value.shape().into(),
            value: valuefact!(self.value.as_ref().clone()),
        };

        Ok((vec![], vec![output]))
    }

    /// Infers properties about the output tensors from the input tensors.
    fn infer_forward(&self, _inputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>> {
        unimplemented!()
    }

    /// Infers properties about the input tensors from the output tensors.
    fn infer_backward(&self, _outputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>> {
        unimplemented!()
    }
}
