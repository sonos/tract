use std::collections::HashMap;
use std::str;
use std::sync::Arc;

pub mod compact;
mod dsl;
mod model;
mod node;
mod order;
mod patch;
mod tensor_info;

pub use self::dsl::*;
pub use self::model::*;
pub use self::node::*;
pub use self::order::eval_order;
pub use self::tensor_info::*;
pub use crate::analyser::types::TensorFact;

use crate::TractResult;

/// A model with partially types and shapes, as produced by parsing ONNX or
/// Tensorflow graphs.
pub type InferenceModel = Model<TensorFact>;
/// Node for InferenceModel graph
pub type InferenceNode = Node<TensorFact>;

/// A model with completely determined types and shapes.
pub type TypedModel = Model<TypedTensorInfo>;
/// Node for TypedModel graph
pub type TypedNode = Node<TypedTensorInfo>;
pub type TypedModelPatch = patch::ModelPatch<TypedTensorInfo>;

/// A model with determined types and shapes, where constant have been
/// eleminated from the graph.
pub type NormalizedModel = Model<NormalizedTensorInfo>;
pub type NormalizedNode = Node<NormalizedTensorInfo>;
pub type NormalizedModelPatch = patch::ModelPatch<NormalizedTensorInfo>;

impl InferenceModel {
    pub fn analyse_one(&mut self, id: usize) -> TractResult<()> {
        crate::analyser::Analyser::new(self)?.analyse_one(id)?;
        Ok(())
    }

    pub fn analyse(&mut self, obstinate: bool) -> TractResult<()> {
        crate::analyser::Analyser::new(self)?.analyse_obstinate(obstinate)
    }

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

    pub fn into_typed(mut self) -> TractResult<TypedModel> {
        self.analyse(false)?;
        compact::compact(&mut self)
    }

    pub fn into_normalized(self) -> TractResult<NormalizedModel> {
        self.into_typed()?.declutter()?.into_normalized()
    }

    pub fn into_optimized(self) -> TractResult<TypedModel> {
        self.into_typed()?.declutter()?.codegen()
    }
}

impl TypedModel {
    pub fn declutter(self) -> TractResult<TypedModel> {
        let mut model = self;
        loop {
            let mut done_something = false;
            for p in crate::optim::declutter() {
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

    pub fn into_normalized(self) -> TractResult<NormalizedModel> {
        compact::compact(&self)
    }

    pub fn into_optimized(self) -> TractResult<TypedModel> {
        let model = self.codegen()?;
        compact::compact(&model)
    }
}

impl NormalizedModel {
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
