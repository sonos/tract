pub use crate::kernels::BinOps;
use crate::ops::MetalEvalOp;
use crate::MetalStream;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;

#[derive(Debug, Clone)]
pub struct MetalBinOp(pub BinOps);

impl MetalBinOp {
    fn resolve_output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let (a, b) = (inputs[0], inputs[1]);
        if a.rank() != b.rank() {
            bail!(
                "Typed ops require rank match. Invalid inputs for {}: {{a: {:?}, b: {:?}}}",
                self.name(),
                a.shape,
                b.shape
            );
        }

        let out_shape = self.0.output_shape(&a.shape, &b.shape)?;
        let out_dt = self.0.output_datum_type(a.datum_type, b.datum_type)?;
        Ok(tvec!(out_dt.fact(out_shape)))
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

crate::impl_eval_op_for_metal_op!(MetalBinOp);

impl MetalEvalOp for MetalBinOp {
    fn metal_eval(
        &self,
        stream: &MetalStream,
        node_id: usize,
        session: &mut SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let (opaque_a, opaque_b) = args_2!(inputs);
        let a = opaque_a.to_device_tensor()?;
        let b = opaque_b.to_device_tensor()?;
        let out_shape = self.0.output_shape(a.shape(), b.shape())?;
        let out_dt = self.0.output_datum_type(a.datum_type(), b.datum_type())?;
        let output = crate::ops::make_tensor_for_node(session, node_id, out_dt, &out_shape)?;
        self.0
            .dispatch_eval(stream, a, b, &output)
            .with_context(|| "Error while dispatching eval for Metal Bin Op")?;

        Ok(tvec!(output.into_opaque_tensor().into_tvalue()))
    }
}

impl TypedOp for MetalBinOp {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| self.resolve_output_facts(facts))
            .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}
