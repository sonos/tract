use crate::internal::*;

#[macro_use]
pub mod macros;
#[macro_use]
pub mod helpers;
#[macro_use]
pub mod rules;

mod analyser;
mod ops;
mod fact;
mod factoid;
mod model;

pub use self::fact::InferenceFact;
pub use self::ops::InferenceOp;
pub use self::factoid::*;

/// A model with partially types and shapes, as produced by parsing ONNX or
/// Tensorflow graphs.
pub type InferenceModel = ModelImpl<InferenceFact, Box<dyn InferenceOp>>;
/// Node for InferenceModel graph
pub type InferenceNode = BaseNode<InferenceFact, Box<dyn InferenceOp>>;
/// A ModelPatch for InferenceModel.
pub type InferenceModelPatch = ModelPatch<InferenceFact, Box<dyn InferenceOp>>;
/// An execution plan for InferenceModel.
pub type InferenceSimplePlan<M> = SimplePlan<InferenceFact, Box<dyn InferenceOp>, M>;
/// An execution state for InferenceModel.
pub type InferenceSimpleState<M, P> = SimpleState<InferenceFact, Box<dyn InferenceOp>, M, P>;

