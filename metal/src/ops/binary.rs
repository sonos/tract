pub use crate::kernels::BinOps;
use crate::IntoMetal;
use tract_core::internal::*;

#[derive(Debug, Clone)]
pub struct MetalBinOp(pub BinOps);

impl Op for MetalBinOp {
    fn name(&self) -> Cow<str> {
        format!("Metal{}", self.0.name()).into()
    }

    fn validation(&self) -> Validation {
        self.0.validation()
    }

    fn same_as(&self, other: &dyn Op) -> bool {
        let Some(other) = other.downcast_ref::<MetalBinOp>() else { return false };
        self.0 == other.0
    }

    op_as_typed_op!();
}

impl EvalOp for MetalBinOp {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let (a, b) = args_2!(inputs);
                let a = a.into_tensor().into_metal()?;
                let b = b.into_tensor().into_metal()?;

                ensure!(a.rank() == b.rank());
                Ok(tvec!(self.0.eval(context, &a, &b)?.into_tensor().into_tvalue()))
            })
        })
    }
}

impl TypedOp for MetalBinOp {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        if inputs[0].rank() != inputs[1].rank() {
            bail!("Typed ops require rank match. Invalid inputs for {}: {:?}", self.name(), inputs);
        }
        let out_dt = self.0.output_datum_type(inputs[0].datum_type, inputs[1].datum_type)?;
        Ok(tvec!(out_dt.fact(&*tract_core::broadcast::multi_broadcast(&[
            &inputs[0].shape.to_tvec(),
            &inputs[1].shape.to_tvec()
        ])?)))
    }

    as_op!();
}
