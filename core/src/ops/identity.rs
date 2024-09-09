use crate::internal::*;

#[derive(Debug, Clone, Default, Hash)]
pub struct Identity;

impl Op for Identity {
    fn name(&self) -> Cow<str> {
        "Identity".into()
    }

    op_as_typed_op!();
}

impl EvalOp for Identity {
    fn is_stateless(&self) -> bool {
        true
    }

    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        Ok(inputs)
    }
}

impl TypedOp for Identity {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].clone()))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        TypedModelPatch::shunt_one_op(model, node)
    }

    fn fuse(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
        TypedModelPatch::shunt_one_op(model, node)
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        AxesMapping::natural(inputs, outputs)
    }

    as_op!();
}

#[derive(Debug, Clone, Default, Hash)]
pub struct PinConst;

impl Op for PinConst {
    fn name(&self) -> Cow<str> {
        "PinConst".into()
    }

    op_as_typed_op!();
}

impl EvalOp for PinConst {
    fn is_stateless(&self) -> bool {
        false
    }

    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        Ok(inputs)
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(self.clone())))
    }
}

impl OpState for PinConst {
    fn eval(
            &mut self,
            _session: &mut SessionState,
            _op: &dyn Op,
            inputs: TVec<TValue>,
        ) -> TractResult<TVec<TValue>> {
        Ok(inputs)
    }
}

impl TypedOp for PinConst {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].without_value()))
    }

    as_op!();
}

trivial_op_state_freeeze!(PinConst);
