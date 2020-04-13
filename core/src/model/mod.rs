//! ## Models and their lifecycle
//!
//! In order to reason on the model and performs optimisations, a model needs
//! to be `typed`. This means all tensor exchanged between the nodes have a
//! well defined element type (f32, i32, etc) and a shape ([1, 12, 53, 13]).
//!
//! A model typically starts as an `InferenceModel`, with minimum or partial
//! tensor type information. At this stage, the application developper can add
//! types and shapes hints (like the model inputs and output element types
//! and shapes), then `tract` will perform type inference propagating this
//! information. Hopefully `tract` will be able to infer a type and shape for
//! all tensors in the model graph.
//!
//! At this stage, the model can be converted into a `TypedModel`.
//!
//! InferanceModel and TypeModel are two variants of `ModelImpl`, Parameterized
//! by a Fact implementation: TypedModel uses TypedFact, enforcing
//! complete determination of element type and shape, and allowing a constant
//! value for the tensor. InferenceModel uses InferenceFact, which can handle
//! partial information.
//!
//! A third type of Model exists and can be useful: NormalizedModel, using
//! NormalizedFact. In this case, constant values are no longer allowed:
//! all tensors exchanged in the network are actual variables.
//! Parts of the graph producing constant values (like constant weights or
//! hyper-parameter computation implemented in the graph) have been
//! eliminated from the graph. This normal form is akin to an IR in compiler
//! technologies. This is the favourite form on which tract optimisation should
//! be implemented.
//!
//! We call `declutter` the process getting the network closer to its normal
//! form: as we just said, constant must be absorbed by the operator that will
//! use them. For instance an Add node adding a constant input to a variable
//! tensor input would be replaced by an unary Add operator taking only the
//! variable input and for which the constant to add is a fixed construction
//! attribute. In the same decluttering process, we try and replace proprietary
//! operators (eg, from TensorFlow) by tract core operators: it is not always
//! possible to simply map TensorFlow operators to tract-core while loading the
//! network: their interfaces can be different (what is an input, what is an
//! attribute) and constant propagation may be necessary before the right
//! core operator could be chosen.
use std::collections::HashMap;
use std::str;

use itertools::Itertools;

pub mod compact;
pub mod dsl;
mod fact;
mod model;
mod node;
pub mod order;
mod patch;
pub mod translator;

pub use self::dsl::*;
pub use self::fact::*;
pub use self::model::*;
pub use self::node::*;
pub use self::order::eval_order;
pub use self::patch::ModelPatch;
pub use crate::ops::{Op, TypedOp};

use crate::model::translator::Translate;
use crate::ops::invariants;
use crate::plan::{SimplePlan, SimpleState};
use crate::TractResult;
use tract_linalg::hash::DynHash;

/// Common methods for all variants of model.
pub trait Model: downcast_rs::Downcast + std::fmt::Debug + dyn_clone::DynClone {
    /// Model label
    fn model_label(&self) -> Option<&str>;

    /// Lookup node id by name
    fn node_id_by_name(&self, name: &str) -> TractResult<usize>;

    /// Node name by id
    fn node_name(&self, id: usize) -> &str;

    /// Node op by id
    fn node_op(&self, id: usize) -> &dyn Op;

    /// Node inputs by id
    fn node_inputs(&self, id: usize) -> &[OutletId];

    /// Number of outputs for a node, by id.
    fn node_output_count(&self, id: usize) -> usize;

    /// Number nodes
    fn nodes_len(&self) -> usize;

    /// Formatted node label
    fn node_display(&self, id: usize) -> String;

    /// Formatted node label
    fn node_debug(&self, id: usize) -> String;

    /// Eval order for the model
    fn eval_order(&self) -> TractResult<Vec<usize>>;

    /// Eval order for the model overriding input and outputs node
    fn eval_order_for_io(&self, inputs: &[usize], outputs: &[usize]) -> TractResult<Vec<usize>>;

    /// Inputs of the model
    fn input_outlets(&self) -> &[OutletId];

    /// Outputs of the model
    fn output_outlets(&self) -> &[OutletId];

    /// Tensorfact for an outlet
    fn outlet_typedfact(&self, outlet: OutletId) -> TractResult<TypedFact>;

    /// Short outlet formatter (id plus fact)
    fn outlet_fact_format(&self, outlet: OutletId) -> String;

    /// Labels for an outlet
    fn outlet_label(&self, id: OutletId) -> Option<&str>;

    /// List consumers of an outlet
    fn outlet_successors(&self, outlet: OutletId) -> &[InletId];
}

impl_downcast!(Model);

/// A model with completely determined types and shapes.
pub type TypedModel = ModelImpl<TypedFact, Box<dyn TypedOp>>;
/// Node for TypedModel graph
pub type TypedNode = BaseNode<TypedFact, Box<dyn TypedOp>>;
/// A ModelPatch for TypedModel.
pub type TypedModelPatch = ModelPatch<TypedFact, Box<dyn TypedOp>>;
/// An execution plan for TypedModel.
pub type TypedSimplePlan<M> = SimplePlan<TypedFact, Box<dyn TypedOp>, M>;
/// An execution state for TypedModel.
pub type TypedSimpleState<M, P> = SimpleState<TypedFact, Box<dyn TypedOp>, M, P>;

/// A model with determined types and shapes, where constant have been
/// eleminated from the graph.
pub type NormalizedModel = ModelImpl<NormalizedFact, Box<dyn TypedOp>>;
/// A Node for NormalizedModel.
pub type NormalizedNode = BaseNode<NormalizedFact, Box<dyn TypedOp>>;
/// A ModelPatch for NormalizedModel.
pub type NormalizedModelPatch = ModelPatch<NormalizedFact, Box<dyn TypedOp>>;
/// An execution plan for NormalizedModel.
pub type NormalizedSimplePlan<M> = SimplePlan<NormalizedFact, Box<dyn TypedOp>, M>;
/// An execution state for TypedModel.
pub type NormalizedSimpleState<M, P> = SimpleState<NormalizedFact, Box<dyn TypedOp>, M, P>;

impl TypedModel {
    /// Perform declutter pass on the network.
    pub fn declutter(self) -> TractResult<TypedModel> {
        let mut model = self;
        let model_inputs = model.input_outlets()?.len();
        let model_outputs = model.output_outlets()?.len();
        loop {
            let mut done_something = false;
            for p in crate::optim::declutter() {
                debug!("done_something {:?} before {:?}", done_something, p);
                done_something = done_something || p.pass(&mut model)?;
                debug!("done_something {:?} after {:?}", done_something, p);
                if cfg!(debug_assertions) {
                    model.check_edges()?;
                    assert_eq!(model.input_outlets()?.len(), model_inputs);
                    assert_eq!(model.output_outlets()?.len(), model_outputs);
                }
            }
            if !done_something {
                break;
            }
            model = compact::compact(&model)?;
        }
        compact::compact(&model)
    }

    /// Translate the graph to optimized operators.
    pub fn codegen(self) -> TractResult<TypedModel> {
        let mut model = self;
        loop {
            let mut done_something = false;
            for p in crate::optim::codegen() {
                done_something = done_something || p.pass(&mut model)?;
                if cfg!(debug_assertions) {
                    model.check_edges()?;
                }
            }
            if !done_something {
                break;
            }
            model = compact::compact(&model)?;
        }
        Ok(model)
    }

    pub fn invariants(&self) -> TractResult<invariants::Invariants> {
        invariants::for_model(self)
    }

    /// Attempt to convert the network to a NormalizedModel.
    pub fn into_normalized(self) -> TractResult<NormalizedModel> {
        crate::model::translator::IntoTranslator.translate_model(&self)
    }

    /// Declutter as much as possible, then translate to optimized operators.
    pub fn into_optimized(self) -> TractResult<TypedModel> {
        let model = self.declutter()?;
        let model = model.codegen()?;
        let model = compact::compact(&model)?;
        Ok(model)
    }
}

impl std::fmt::Display for &dyn Model {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        for i in 0..self.nodes_len() {
            let input_1 =
                self.node_inputs(i).get(0).map(|o| format!("{:?}", o)).unwrap_or("".to_string());
            let input_2 =
                self.node_inputs(i).get(1).map(|o| format!("{:?}", o)).unwrap_or("".to_string());
            let output_1 = self
                .outlet_successors(OutletId::new(i, 0))
                .get(0)
                .map(|o| format!("{:?}", o))
                .unwrap_or("".to_string());
            let output_2 = self
                .outlet_successors(OutletId::new(i, 0))
                .get(1)
                .map(|o| format!("{:?}", o))
                .unwrap_or("".to_string());
            writeln!(
                fmt,
                "{:8} {:8} -> {:5} -> {:8} {:8} | {:15} {}",
                input_1,
                input_2,
                i,
                output_1,
                output_2,
                self.node_op(i).name(),
                self.node_name(i),
            )?;
            if self.node_inputs(i).len() > 2 {
                writeln!(
                    fmt,
                    "                                                | * inputs: {}",
                    self.node_inputs(i).iter().map(|s| format!("{:?}", s)).join(", ")
                )?;
            }
            if self.node_output_count(i) > 1 || self.outlet_successors((i, 0).into()).len() > 2 {
                for o in 0..self.node_output_count(i) {
                    writeln!(
                        fmt,
                        "                                                | * output #{}: {}",
                        o,
                        self.outlet_successors((i, o).into())
                            .iter()
                            .map(|s| format!("{:?}", s))
                            .join(", ")
                    )?;
                }
            }
        }
        Ok(())
    }
}

impl NormalizedModel {
    /// Convert back to TypedModel.
    ///
    /// Can not fail.
    pub fn into_typed(self) -> TractResult<TypedModel> {
        translator::IntoTranslator.translate_model(&self)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test() {
        fn is_sync<T: Sync>() {}
        is_sync::<TypedModel>();
        is_sync::<NormalizedModel>();
    }
}
