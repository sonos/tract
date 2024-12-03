use crate::ops::{MetalAxisOp, MetalEvalOp, MetalOpState};
use crate::tensor::MetalTensorExt;
use crate::MetalContext;
use derive_new::new;
use tract_core::internal::*;

#[derive(Clone, Debug, new, Hash)]
pub struct FusedAxisOp<O: MetalEvalOp + TypedOp> {
    pub axis_ops: Vec<Option<MetalAxisOp>>,
    pub op: O,
}

impl<O: MetalEvalOp + TypedOp> Op for FusedAxisOp<O> {
    fn name(&self) -> Cow<str> {
        self.op.name()
    }

    op_as_typed_op!();
}

impl<O: MetalEvalOp + TypedOp> MetalEvalOp for FusedAxisOp<O> {
    fn metal_eval(
        &self,
        context: &MetalContext,
        node_id: usize,
        session: &mut SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        // Apply Axis Op
        let inputs = inputs
            .into_iter()
            .zip(self.axis_ops.iter())
            .map(|(input, axis_op)| {
                let Some(axis_op) = axis_op else { return Ok(input) };
                let new_shape = match &axis_op.0 {
                    AxisOp::Move(..) => bail!("Cannot fused {:?} with metal op", &axis_op.0),
                    AxisOp::Reshape(skip, from, to) => {
                        let from = from.iter().map(|d| d.eval(&session.resolved_symbols)).collect();
                        let to = to.iter().map(|d| d.eval(&session.resolved_symbols)).collect();
                        let mut shape: TVec<usize> = input.shape().into();
                        AxisOp::Reshape(*skip, from, to).change_shape_array(&mut shape, false)?;
                        shape
                    }
                    _ => {
                        let mut shape: TVec<usize> = input.shape().into();
                        axis_op.0.change_shape_array(&mut shape, false)?;
                        shape
                    }
                };
                let t = input.to_metal_tensor()?;
                Ok(t.reshaped(new_shape)?.into_opaque_tensor().into())
            })
            .collect::<TractResult<TVec<_>>>()?;
        self.op.metal_eval(context, node_id, session, inputs)
    }
}

impl<O: MetalEvalOp + TypedOp> TypedOp for FusedAxisOp<O> {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(
            inputs.len() == self.axis_ops.len(),
            "Number of inputs and fused axis ops are not aligned"
        );
        // Apply AxisOp
        let inputs = inputs
            .into_iter()
            .zip(self.axis_ops.iter())
            .map(|(i, axis_op)| {
                Ok(axis_op
                    .as_ref()
                    .map(|a| -> TractResult<_> { Ok(a.output_facts(&[i])?.pop()) })
                    .transpose()?
                    .flatten()
                    .unwrap_or_else(|| (*i).clone()))
            })
            .collect::<TractResult<TVec<_>>>()?;
        let inputs_ref = inputs.iter().collect::<TVec<_>>();
        // Apply Op
        self.op.output_facts(&inputs_ref)
    }

    as_op!();
}

impl<O: MetalEvalOp + TypedOp> EvalOp for FusedAxisOp<O> {
    fn is_stateless(&self) -> bool {
        false
    }

    #[allow(unused_variables)]
    fn state(
        &self,
        session: &mut tract_core::internal::SessionState,
        node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(MetalOpState::new(node_id, self.clone()))))
    }
}
