use tract_hir::internal::*;

use crate::model::ParsingContext;
use crate::pb::NodeProto;

pub fn non_zero(
    ctx: &ParsingContext,
    _node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    // symbol table is shared between all templates and models
    let count = ctx.template.symbols.new_with_prefix("x");
    Ok((Box::new(NonZero { count }) as _, vec![]))
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct NonZero {
    count: Symbol,
}

impl NonZero {
    fn typed(&self) -> tract_onnx_opl::non_zero::NonZero {
        tract_onnx_opl::non_zero::NonZero { count: self.count.clone() }
    }
}

impl Op for NonZero {
    fn name(&self) -> StaticName {
        "NonZero".into()
    }

    not_a_typed_op!();
}

impl EvalOp for NonZero {
    fn is_pure_function(&self) -> bool {
        true
    }

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        self.typed().eval(_ctx, inputs)
    }
}

impl InferenceRulesOp for NonZero {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> TractResult<()> {
        check_input_arity(inputs, 1)?;
        check_output_arity(outputs, 1)?;
        s.equals(&outputs[0].datum_type, i64::datum_type())?;
        s.equals(&outputs[0].rank, 2)?;
        s.equals(&outputs[0].shape[0], inputs[0].rank.bex().to_dim())?;
        Ok(())
    }

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &std::collections::HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let inputs = node.inputs.iter().map(|o| mapping[o]).collect::<TVec<_>>();
        target.wire_node(&node.name, self.typed(), &inputs)
    }

    as_op!();
}
