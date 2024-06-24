use crate::internal::*;

#[derive(Debug, Clone, new, Hash)]
pub struct MetalMultiBroadcastTo {
    pub shape: ShapeFact,
}

impl Op for MetalMultiBroadcastTo {
    fn name(&self) -> Cow<str> {
        "MetalMultiBroadcastTo".into()
    }

    op_as_typed_op!();
}

impl EvalOp for MetalMultiBroadcastTo {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        session: &SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {

        let shape = self.shape.eval_to_usize(&session.resolved_symbols)?;
        let input = args_1!(inputs);
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let input = input.into_tensor().into_metal()?;
                Ok(tvec![kernels::MultiBroadcast
                    .eval(context, &input, shape)?
                    .to_cpu()
                    .into_tvalue()])
            })
        })
    }
}

impl TypedOp for MetalMultiBroadcastTo {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = inputs[0].datum_type.fact(self.shape.clone());
        fact.uniform.clone_from(&inputs[0].uniform);
        Ok(tvec!(fact))
    }

    as_op!();
}
