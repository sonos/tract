use std::fmt::Debug;

use tract_downcast_rs::Downcast;

use crate::internal::*;

#[derive(Debug, Clone)]
pub struct SubmodelOp {
    pub model: Box<dyn InnerModel>,
    label: String,
    decluttered: bool,
    codegen: bool,
}

impl PartialEq for SubmodelOp {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}
impl Eq for SubmodelOp {}

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
    fn name(&self) -> StaticName {
        "SubmodelOp".into()
    }

    op_as_typed_op!();
}

impl EvalOp for SubmodelOp {
    not_out_of_plan!();

    fn state(&self, ctx: &EvalContext) -> TractResult<Option<Box<dyn OpState>>> {
        self.model.state(ctx)
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

    #[allow(unused_variables)]
    fn state(&self, ctx: &EvalContext) -> TractResult<Option<Box<dyn OpState>>> {
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
    #[allow(unused_variables)]
    fn state(&self, ctx: &EvalContext) -> TractResult<Option<Box<dyn OpState>>> {
        let plan = self.clone().into_runnable()?;
        let state = plan.spawn()?;
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

pub type TypedModelOpState = TypedSimpleState;

impl OpState for TypedModelOpState {
    fn eval(
        &mut self,
        _ctx: &EvalContext,
        _op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let inference_out = self.run(inputs)?;
        Ok(inference_out)
    }

    fn reset_lanes(&mut self, _lanes: &[LaneId]) -> TractResult<()> {
        bail!("Submodel is not lane-aware: its body is a nested state")
    }
}
