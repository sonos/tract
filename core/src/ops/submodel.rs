use crate::{internal::*, ops::OpStateFreeze};

#[derive(Debug, Clone)]
pub struct SubmodelOp {
    model: TypedModel,
    label: String,
    decluttered: bool,
    optimized: bool,
}

impl SubmodelOp {
    pub fn new(model: TypedModel, label: &str) -> TractResult<Self> {
        Ok(Self { model, label: label.to_string(), decluttered: false, optimized: false })
    }

    pub fn model(&self) -> &TypedModel {
        &self.model
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
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(SubmodelOpState::new(self)?)))
    }
}

impl TypedOp for SubmodelOp {
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let facts = self
            .model
            .output_outlets()?
            .into_iter()
            .map(|outlet| self.model.outlet_fact(*outlet).map(|c| c.clone()))
            .collect::<TractResult<TVec<_>>>()?;
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
        if !self.optimized {
            let mut new = self.clone();
            new.model.optimize()?;
            new.optimized = true;
            Ok(Some(TypedModelPatch::replace_single_op(model, node, &node.inputs, new)?))
        } else {
            Ok(None)
        }
    }

    as_op!();
}

#[derive(Debug, Clone)]
pub struct SubmodelOpState {
    state: TypedSimpleState<TypedModel, TypedSimplePlan<TypedModel>>,
}

impl SubmodelOpState {
    pub fn new(op: &SubmodelOp) -> TractResult<Self> {
        let plan = SimplePlan::new(op.model.clone())?;
        let state = SimpleState::new(plan)?;
        Ok(Self { state })
    }
}

impl OpState for SubmodelOpState {
    fn eval(
        &mut self,
        _session: &mut SessionState,
        _op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let inference_out = self.state.run(inputs)?;
        Ok(inference_out)
    }
}

#[derive(Debug, Clone)]
pub struct FrozenSubmodelOpState {
    state: TypedFrozenSimpleState<TypedModel, TypedSimplePlan<TypedModel>>,
}

impl FrozenOpState for FrozenSubmodelOpState {
    fn unfreeze(&self) -> Box<dyn OpState> {
        Box::new(SubmodelOpState { state: self.state.unfreeze() })
    }
}

impl OpStateFreeze for SubmodelOpState {
    fn freeze(&self) -> Box<dyn FrozenOpState> {
        Box::new(FrozenSubmodelOpState { state: self.state.freeze() })
    }
}
