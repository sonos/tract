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

pub use tract_core::dim::TDim;

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

#[derive(Clone, Debug)]
pub struct InferenceModelChecker;
impl ModelChecker<InferenceFact, Box<dyn InferenceOp>> for InferenceModelChecker {
    fn check(model: &InferenceModel) -> TractResult<()> {
        for node in model.nodes() {
            let (inputs, outputs) = model.node_facts(node.id)?;
            let observed: TVec<&InferenceFact> = node
                .op
                .observe_outlets(&model, &node)?
                .iter()
                .map(|o| model.outlet_fact(*o))
                .collect::<TractResult<_>>()?;
            // may be hugely expensive but this code is not meant to run in release
            let mut op = node.op.clone();
            op.infer_facts(inputs, outputs, observed)?;
        }
        Ok(())
    }
}

/// A model with partially types and shapes, as produced by parsing ONNX or
/// Tensorflow graphs.
pub type InferenceModel = ModelImpl<InferenceFact, Box<dyn InferenceOp>, InferenceModelChecker>;
/// Node for InferenceModel graph
pub type InferenceNode = BaseNode<InferenceFact, Box<dyn InferenceOp>>;
/// A ModelPatch for InferenceModel.
pub type InferenceModelPatch =
    ModelPatch<InferenceFact, Box<dyn InferenceOp>, InferenceModelChecker>;
/// An execution plan for InferenceModel.
pub type InferenceSimplePlan<M> =
    SimplePlan<InferenceFact, Box<dyn InferenceOp>, InferenceModelChecker, M>;
/// An execution state for InferenceModel.
pub type InferenceSimpleState<M, P> =
    SimpleState<InferenceFact, Box<dyn InferenceOp>, InferenceModelChecker, M, P>;
