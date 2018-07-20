use std::collections::HashMap;

use super::{Attr, Op, OpRegister, TensorView};
use std::sync::Arc;
use tfpb::types::DataType;
use {Result, Tensor};
use analyser::interface::*;

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
}

impl ::ops::InferenceRulesOp for Const {
    fn rules<'r, 'p: 'r>(&self, solver: &mut Solver<'r>, inputs: &'p TensorsProxy, outputs: &'p TensorsProxy) {
        // infer will call eval as "all" inputs are known
        solver
            .equals(&inputs.len, 0)
            .equals(&outputs.len, 1);
    }
}
