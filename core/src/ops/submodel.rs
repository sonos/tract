use std::fmt::Debug;

use tract_downcast_rs::Downcast;

use crate::{internal::*, ops::OpStateFreeze};

#[derive(Debug, Clone)]
pub struct SubmodelOp {
    pub model: Box<dyn InnerModel>,
    label: String,
    decluttered: bool,
    codegen: bool,
}

impl SubmodelOp {
    pub fn new(model: Box<dyn InnerModel>, label: &str) -> TractResult<Self> {
        Ok(Self { model, label: label.to_string(), decluttered: false, codegen: false })
    }

    pub fn iteration_count(&self, _inputs: &[&TypedFact]) -> Option<TDim> {
        None
    }

    pub fn model(&self) -> &TypedModel {
        self.model.as_typed()
    }

    pub fn label(&self) -> &str {
        self.label.as_str()
    }
}

impl Op for SubmodelOp {
    fn name(&self) -> Cow<str> {
        "SubmodelOp".into()
    }

    op_as_typed_op!();
}

impl EvalOp for SubmodelOp {
    fn is_stateless(&self) -> bool {
        false
    }

    fn state(
        &self,
        session: &mut SessionState,
        node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        self.model.state(session, node_id)
    }
}

impl TypedOp for SubmodelOp {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let facts = self.model.output_facts(inputs)?;
        Ok(facts)
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if !self.decluttered {
            let mut new = self.clone();
            new.model.declutter()?;
            new.decluttered = true;
            Ok(Some(TypedModelPatch::replace_single_op(model, node, &node.inputs, new)?))
        } else {
            Ok(None)
        }
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if !self.codegen {
            let mut new = self.clone();
            new.model.codegen()?;
            new.codegen = true;
            Ok(Some(TypedModelPatch::replace_single_op(model, node, &node.inputs, new)?))
        } else {
            Ok(None)
        }
    }

    as_op!();
}

pub trait InnerModel: Debug + dyn_clone::DynClone + Downcast + Sync + Send + 'static {
    #[allow(unused_variables)]
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>>;
    fn is_stateless(&self) -> bool;

    #[allow(unused_variables)]
    fn state(
        &self,
        session: &mut SessionState,
        node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(None)
    }

    #[allow(unused_variables)]
    fn declutter(&mut self) -> TractResult<()>;

    fn codegen(&mut self) -> TractResult<()>;

    fn as_typed(&self) -> &TypedModel;
}

dyn_clone::clone_trait_object!(InnerModel);
downcast_rs::impl_downcast!(InnerModel);

impl InnerModel for TypedModel {
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let facts = self
            .output_outlets()?
            .iter()
            .map(|outlet| self.outlet_fact(*outlet).cloned())
            .collect::<TractResult<TVec<_>>>()?;
        Ok(facts)
    }
    fn is_stateless(&self) -> bool {
        false
    }

    #[allow(unused_variables)]
    fn state(
        &self,
        session: &mut SessionState,
        node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        let plan = SimplePlan::new(self.clone())?;
        let state = SimpleState::new(Arc::new(plan))?;
        Ok(Some(Box::new(state)))
    }

    #[allow(unused_variables)]
    fn declutter(&mut self) -> TractResult<()> {
        self.declutter()
    }

    fn codegen(&mut self) -> TractResult<()> {
        self.optimize()
    }

    fn as_typed(&self) -> &TypedModel {
        self
    }
}

pub type TypedModelOpState = TypedSimpleState<TypedModel, Arc<TypedSimplePlan<TypedModel>>>;

impl OpState for TypedModelOpState {
    fn eval(
        &mut self,
        _session: &mut SessionState,
        _op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let inference_out = self.run(inputs)?;
        Ok(inference_out)
    }
}

pub type FrozenSubmodelOpState =
    TypedFrozenSimpleState<TypedModel, Arc<TypedSimplePlan<TypedModel>>>;

impl FrozenOpState for FrozenSubmodelOpState {
    fn unfreeze(&self) -> Box<dyn OpState> {
        Box::new(self.unfreeze())
    }
}

impl OpStateFreeze for TypedModelOpState {
    fn freeze(&self) -> Box<dyn FrozenOpState> {
        Box::new(self.freeze())
    }
}
