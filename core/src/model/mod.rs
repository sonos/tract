//! B
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
//! by a TensorInfo implementation: TypedModel uses TypedTensorInfo, enforcing
//! complete determination of element type and shape, and allowing a constant
//! value for the tensor. InferenceModel uses TensorFact, which can handle
//! partial information.
//!
//! A third type of Model exists and can be useful: NormalizedModel, using
//! NormalizedTensorInfo. In this case, constant values are no longer allowed:
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

pub(crate) mod compact;
mod dsl;
mod model;
mod node;
pub mod order;
mod patch;
mod tensor_info;

pub use self::dsl::*;
pub use self::model::*;
pub use self::node::*;
pub use self::order::eval_order;
pub use self::patch::ModelPatch;
pub use self::tensor_info::*;
pub use crate::analyser::types::TensorFact;
pub use crate::ops::{InferenceOp, Op, TypedOp};

use crate::TractResult;
use crate::plan::{ SimplePlan, SimpleState };

/// Common methods for all variants of model.
pub trait Model: downcast_rs::Downcast + std::fmt::Debug + objekt::Clone {
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

    /// Number of outputs for a node, by id.
    fn node_control_inputs(&self, id: usize) -> &[usize];

    /// Number nodes
    fn nodes_len(&self) -> usize;

    /// Formatted node label
    fn node_format(&self, id: usize) -> String;

    /// Eval order for the model
    fn eval_order(&self) -> TractResult<Vec<usize>>;

    /// Eval order for the model overriding input and outputs node
    fn eval_order_for_io(&self, inputs: &[usize], outputs: &[usize]) -> TractResult<Vec<usize>>;

    /// Inputs of the model
    fn input_outlets(&self) -> &[OutletId];

    /// Outputs of the model
    fn output_outlets(&self) -> &[OutletId];

    /// Tensorfact for an outlet
    fn outlet_tensorfact(&self, outlet: OutletId) -> TensorFact;

    /// Tensorfact for an outlet
    fn outlet_fact_format(&self, outlet: OutletId) -> String;

    /// List consumers of an outlet
    fn outlet_successors(&self, outlet: OutletId) -> &[InletId];
}

impl_downcast!(Model);

#[macro_export]
macro_rules! dispatch_model {
    ($model: expr, $expr: expr) => {
        if let Some(m) = $model.downcast_ref::<InferenceModel>() {
            $expr(m)
        } else if let Some(m) = $model.downcast_ref::<TypedModel>() {
            $expr(m)
        } else if let Some(m) = $model.downcast_ref::<NormalizedModel>() {
            $expr(m)
        } else if let Some(m) = $model.downcast_ref::<PulsedModel>() {
            $expr(m)
        } else {
            unreachable!()
        }
    }
}

#[macro_export]
macro_rules! dispatch_model_no_pulse {
    ($model: expr, $expr: expr) => {
        if let Some(m) = $model.downcast_ref::<InferenceModel>() {
            $expr(m)
        } else if let Some(m) = $model.downcast_ref::<TypedModel>() {
            $expr(m)
        } else if let Some(m) = $model.downcast_ref::<NormalizedModel>() {
            $expr(m)
        } else {
            bail!("Pulse model are unsupported here")
        }
    }
}

/// A model with partially types and shapes, as produced by parsing ONNX or
/// Tensorflow graphs.
pub type InferenceModel = ModelImpl<TensorFact, Box<dyn InferenceOp>>;
/// Node for InferenceModel graph
pub type InferenceNode = BaseNode<TensorFact, Box<dyn InferenceOp>>;
/// A ModelPatch for InferenceModel.
pub type InferenceModelPatch = ModelPatch<TensorFact, Box<dyn InferenceOp>>;
/// An execution plan for InferenceModel.
pub type InferenceSimplePlan<M> = SimplePlan<TensorFact, Box<dyn InferenceOp>, M>;
/// An execution state for InferenceModel.
pub type InferenceSimpleState<M,P> = SimpleState<TensorFact, Box<dyn InferenceOp>, M, P>;

/// A model with completely determined types and shapes.
pub type TypedModel = ModelImpl<TypedTensorInfo, Box<dyn TypedOp>>;
/// Node for TypedModel graph
pub type TypedNode = BaseNode<TypedTensorInfo, Box<dyn TypedOp>>;
/// A ModelPatch for TypedModel.
pub type TypedModelPatch = ModelPatch<TypedTensorInfo, Box<dyn TypedOp>>;
/// An execution plan for TypedModel.
pub type TypedSimplePlan<M> = SimplePlan<TypedTensorInfo, Box<dyn TypedOp>, M>;
/// An execution state for TypedModel.
pub type TypedSimpleState<M,P> = SimpleState<TypedTensorInfo, Box<dyn TypedOp>, M, P>;

/// A model with determined types and shapes, where constant have been
/// eleminated from the graph.
pub type NormalizedModel = ModelImpl<NormalizedTensorInfo, Box<dyn TypedOp>>;
/// A Node for NormalizedModel.
pub type NormalizedNode = BaseNode<NormalizedTensorInfo, Box<dyn TypedOp>>;
/// A ModelPatch for NormalizedModel.
pub type NormalizedModelPatch = ModelPatch<NormalizedTensorInfo, Box<dyn TypedOp>>;
/// An execution plan for NormalizedModel.
pub type NormalizedSimplePlan<M> = SimplePlan<NormalizedTensorInfo, Box<dyn TypedOp>, M>;
/// An execution state for TypedModel.
pub type NormalizedSimpleState<M,P> = SimpleState<NormalizedTensorInfo, Box<dyn TypedOp>, M, P>;

impl InferenceModel {
    /// Analyse one node of the graph.
    pub fn analyse_one(&mut self, id: usize) -> TractResult<()> {
        crate::analyser::Analyser::new(self).analyse_one(id)?;
        Ok(())
    }

    /// Analyse all nodes of the graph.
    ///
    /// Will stop on first error unless `obstinate` is `true`.
    pub fn analyse(&mut self, obstinate: bool) -> TractResult<()> {
        crate::analyser::Analyser::new(self).analyse_obstinate(obstinate)
    }

    /// Perform early transformation before going typed.
    pub fn incorporate(self) -> TractResult<InferenceModel> {
        let mut model = self;
        loop {
            let mut done_something = false;
            for p in crate::optim::incorporate() {
                done_something = done_something || p.pass(&mut model)?;
                if cfg!(debug_assertions) {
                    model.check_edges()?;
                }
            }
            if !done_something {
                break;
            }
        }
        model = compact::compact(&model)?;
        model.analyse(false)?;
        Ok(model)
    }

    /// List OutletId with incomplete type information.
    ///
    /// Will stop on first error unless `obstinate` is `true`.
    pub fn missing_type_shape(&self) -> TractResult<Vec<OutletId>> {
        use crate::analyser::types::Fact;
        Ok(self
            .eval_order()?
            .iter()
            .flat_map(|&node| {
                self.nodes()[node]
                    .outputs
                    .iter()
                    .enumerate()
                    .map(move |(ix, outlet)| (OutletId::new(node, ix), outlet))
            })
            .filter(|(_, o)| !o.fact.datum_type.is_concrete() || !o.fact.shape.is_concrete())
            .map(|(id, _)| id)
            .collect())
    }

    /// Eliminate seemingly dead branches of the graph.
    ///
    /// This may break stateful networks.
    pub fn eliminate_dead_branches(mut self) -> TractResult<InferenceModel> {
        compact::compact(&mut self)
    }

    /// Attempt full analyse and conversion to TypedModel.
    pub fn into_typed(mut self) -> TractResult<TypedModel> {
        self.analyse(false)?;
        let m = self.incorporate()?;
        Ok(compact::translate(&m, &())?.0)
    }

    /// Attempt full analyse, decluttering and conversion to NormalizedModel.
    pub fn into_normalized(self) -> TractResult<NormalizedModel> {
        self.into_typed()?.declutter()?.into_normalized()
    }

    /// Attempt full analyse, decluttering and mapping to optimized operations.
    ///
    /// This will work even if the network can not be normalized.
    pub fn into_optimized(self) -> TractResult<TypedModel> {
        self.into_typed()?.declutter()?.codegen()
    }
}

impl TypedModel {
    /// Perform declutter pass on the network.
    pub fn declutter(self) -> TractResult<TypedModel> {
        let mut model = self;
        let model_inputs = model.input_outlets()?.len();
        let model_outputs = model.output_outlets()?.len();
        loop {
            let mut done_something = false;
            for p in crate::optim::declutter() {
                done_something = done_something || p.pass(&mut model)?;
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
        Ok(model)
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

    /// Attempt to convert the network to a NormalizedModel.
    pub fn into_normalized(self) -> TractResult<NormalizedModel> {
        compact::compact(&self)
    }

    /// Declutter as much as possible, then translate to optimized operators.
    pub fn into_optimized(self) -> TractResult<TypedModel> {
        let model = self.declutter()?;
        let model = model.codegen()?;
        let model = compact::compact(&model)?;
        Ok(model)
    }
}

impl NormalizedModel {
    /// Convert back to TypedModel.
    ///
    /// Can not fail.
    pub fn into_typed(self) -> TractResult<TypedModel> {
        compact::compact(&self)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test() {
        fn is_sync<T: Sync>() {}
        is_sync::<InferenceModel>();
        is_sync::<TypedModel>();
        is_sync::<NormalizedModel>();
    }
}
