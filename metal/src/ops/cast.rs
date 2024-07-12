use crate::kernels;
use crate::tensor::MetalTensorExt;
use derive_new::new;
use tract_core::internal::*;

#[derive(Debug, Clone, new, Hash, PartialEq, Eq)]
pub struct MetalCast {
    pub to: DatumType,
}

impl Op for MetalCast {
    fn name(&self) -> Cow<str> {
        "MetalCast".into()
    }

    op_as_typed_op!();
    impl_op_same_as!();
}

impl EvalOp for MetalCast {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let t = input.to_metal_tensor()?;
        if t.datum_type() == self.to {
            Ok(tvec!(input))
        } else {
            objc::rc::autoreleasepool(|| {
                crate::METAL_CONTEXT.with_borrow(|context| {
                    Ok(tvec![kernels::array::Cast
                        .dispatch_eval(context, t, self.to)?
                        .into_opaque_tensor()
                        .into_tvalue()])
                })
            })
        }
    }
}

impl TypedOp for MetalCast {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(self.to.fact(inputs[0].shape.clone())))
    }

    as_op!();
}
