use std::collections::HashMap;

use super::{Attr, Op, OpRegister, Value};
use analyser::rules::prelude::*;
use {DatumType, Result, Tensor};

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("Const", Const::build);
}

#[derive(Debug, Clone)]
pub struct Const {
    dtype: DatumType,
    value: Value,
}

impl Const {
    pub fn for_tensor(tensor: Tensor) -> Const {
        let dtype = tensor.datum_type();
        let value: Value = tensor.into();
        Const {
            dtype,
            value: value.into_shared(),
        }
    }
    pub fn build(node: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        let dtype = node.get_attr_datum_type("dtype")?;
        let mat = node.get_attr_tensor("value")?;

        if mat.datum_type() != dtype {
            bail!(
                "Const node {:?} doesn't have the expected {:?} type.",
                mat,
                dtype
            );
        }

        Ok(Box::new(Const {
            dtype,
            value: Value::from(mat).into_shared(),
        }))
    }
}

impl Op for Const {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, _inputs: TVec<Value>) -> Result<TVec<Value>> {
        Ok(tvec![self.value.clone()])
    }

    /// Returns the attributes of the operation and their values.
    fn get_attributes(&self) -> HashMap<&'static str, Attr> {
        hashmap!{
            "dtype" => Attr::DatumType(self.dtype),
            "value" => Attr::Tensor(self.value.as_tensor().clone()),
        }
    }

    fn const_value(&self) -> Option<Value> {
        Some(self.value.clone())
    }
}

impl InferenceRulesOp for Const {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) {
        solver.equals(&inputs.len, 0).equals(&outputs.len, 1);
    }
}
