use crate::internal::*;

#[macro_use]
pub mod helpers;
#[macro_use]
pub mod rules;

mod analyser;
mod fact;
mod factoid;
mod model;
mod ops;
mod optim;

pub use self::fact::InferenceFact;
pub use self::factoid::*;
pub use self::model::InferenceModelExt;
pub use self::ops::InferenceOp;
pub use self::rules::expr::IntoExp;
pub use self::rules::expr::ToDimExp;
pub use self::rules::InferenceResult;
pub use self::rules::InferenceRulesOp;
pub use self::rules::Solver;
pub use self::rules::TensorProxy;
pub use wrap;

pub fn check_input_arity(inputs: &[TensorProxy], expected: usize) -> TractResult<()> {
    if inputs.len() != expected {
        bail!("Wrong input number. Rules expect {}, node has {}.", expected, inputs.len())
    } else {
        Ok(())
    }
}

pub fn check_output_arity(outputs: &[TensorProxy], expected: usize) -> TractResult<()> {
    if outputs.len() != expected {
        bail!("Wrong output number. Rules expect {}, node has {}.", expected, outputs.len())
    } else {
        Ok(())
    }
}

/// A model with partially types and shapes, as produced by parsing ONNX or
/// Tensorflow graphs.
pub type InferenceModel = Graph<InferenceFact, Box<dyn InferenceOp>>;
/// Node for InferenceModel graph
pub type InferenceNode = Node<InferenceFact, Box<dyn InferenceOp>>;
/// A ModelPatch for InferenceModel.
pub type InferenceModelPatch = ModelPatch<InferenceFact, Box<dyn InferenceOp>>;
/// An execution plan for InferenceModel.
pub type InferenceSimplePlan<M> = SimplePlan<InferenceFact, Box<dyn InferenceOp>, M>;
/// An execution state for InferenceModel.
pub type InferenceSimpleState<M, P> = SimpleState<InferenceFact, Box<dyn InferenceOp>, M, P>;

impl<'a> From<&'a Box<dyn InferenceOp>> for Box<dyn InferenceOp> {
    fn from(it: &'a Box<dyn InferenceOp>) -> Box<dyn InferenceOp> {
        tract_core::dyn_clone::clone_box(it.as_ref())
    }
}
