use tract_core::ops::prelude::*;
use std::collections::HashMap;

use crate::tfpb::node_def::NodeDef;
use crate::model::TfOpRegister;

#[macro_use]
mod macros;

pub mod array;
pub mod logic;
pub mod math;
pub mod nn;
pub mod quant;

pub fn register_all_ops(reg: &mut TfOpRegister) {
    array::register_all_ops(reg);
    logic::register_all_ops(reg);
    math::register_all_ops(reg);
    nn::register_all_ops(reg);
    quant::register_all_ops(reg);
    reg.insert("Const", konst);
    reg.insert("NoOp", |_| Ok(Box::new(Noop)));
    reg.insert("Placeholder", placeholder);
}


pub fn konst(node: &NodeDef) -> TractResult<Box<Op>> {
    let dtype = node.get_attr_datum_type("dtype")?;
    let mat = node.get_attr_tensor("value")?;

    if mat.datum_type() != dtype {
        bail!(
            "Const node {:?} doesn't have the expected {:?} type.",
            mat,
            dtype
        );
    }

    Ok(Box::new(::tract_core::ops::konst::Const::for_tensor(mat)))
}

pub fn placeholder(node: &NodeDef) -> TractResult<Box<Op>> {
    let dt = node.get_attr_datum_type("dtype")?;
    let mut fact = TensorFact::dt(dt);
    if let Some(shape) = node.get_attr_opt_shape("shape")? {
        fact = fact.with_shape(shape)
    }
    Ok(Box::new(::tract_core::ops::source::Source::new(fact)))
}

#[derive(Clone, Debug, new)]
struct Noop;

impl Op for Noop {
    fn name(&self) -> Cow<str> {
        "tf.Noop".into()
    }
}

impl StatelessOp for Noop {
    fn eval(&self, _inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        Ok(tvec!())
    }
}

impl InferenceRulesOp for Noop {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        _s: &mut Solver<'r>,
        _inputs: &'p [TensorProxy],
        _outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        Ok(())
    }
}
