pub use crate::kernels::BinOps;
use crate::tensor::MetalTensorExt;
use tract_core::internal::*;

#[derive(Debug, Clone)]
pub struct MetalBinOp(pub BinOps);

impl MetalBinOp {
    fn resolve_output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let (a, b) = (inputs[0], inputs[1]);
        if a.rank() != b.rank() {
            bail!("Typed ops require rank match. Invalid inputs for {}: {:?}", self.name(), inputs);
        }
        let out_dt = self.0.output_datum_type(a.datum_type, b.datum_type)?;
        Ok(tvec!(out_dt.fact(&*tract_core::broadcast::multi_broadcast(&[
            &a.shape.to_tvec(),
            &b.shape.to_tvec()
        ])?)))
    }
}

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
                let a_metal = a.to_metal_tensor()?;
                let b_metal = b.to_metal_tensor()?;
                ensure!(a.rank() == b.rank());
                Ok(tvec!(self
                    .0
                    .dispatch_eval(context, a_metal, b_metal)
                    .with_context(|| "Error while dispatching eval for Metal Bin Op")?
                    .into_opaque_tensor()
                    .into_tvalue()))
            })
        })
    }
}

impl TypedOp for MetalBinOp {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::metal_output_facts(inputs, |facts| self.resolve_output_facts(facts))
            .with_context(|| anyhow::anyhow!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}
