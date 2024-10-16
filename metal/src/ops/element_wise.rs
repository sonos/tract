pub use crate::kernels::ElementWiseOps;
use crate::{MetalTensor, MetalTensorExt};
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
                let opaque_a = args_1!(inputs);
                let a = opaque_a.to_metal_tensor()?;
                let output = unsafe { MetalTensor::uninitialized_dt(a.datum_type(), a.shape())? };
                self.0.dispatch_eval(context, a, &output)?;
                Ok(tvec![output.into_opaque_tensor().into_tvalue()])
            })
        })
    }
}

impl TypedOp for MetalElementWiseOp {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::metal_tmp_output_facts(inputs, |facts| Ok(tvec!(facts[0].without_value())))
            .with_context(|| anyhow::anyhow!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}
