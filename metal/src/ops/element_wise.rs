pub use crate::kernels::ElementWiseOps;
use crate::IntoMetal;
use tract_core::internal::*;

#[derive(Debug, Clone)]
pub struct MetalElementWiseOp(pub ElementWiseOps);

impl Op for MetalElementWiseOp {
    fn name(&self) -> Cow<str> {
        format!("Metal{}", self.0.name()).into()
    }

    fn validation(&self) -> Validation {
        self.0.validation()
    }

    fn same_as(&self, other: &dyn Op) -> bool {
        let Some(other) = other.downcast_ref::<MetalElementWiseOp>() else { return false };
        self.0 == other.0
    }

    op_as_typed_op!();
}

impl EvalOp for MetalElementWiseOp {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let a = args_1!(inputs);
                let a = a.into_tensor().into_metal()?;
                Ok(tvec!(self.0.eval(context, &a)?.into_tensor().into_tvalue()))
            })
        })
    }
}

impl TypedOp for MetalElementWiseOp {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec![inputs[0].clone().without_value()])
    }

    as_op!();
}
