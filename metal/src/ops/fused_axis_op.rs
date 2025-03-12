use crate::ops::{MetalAxisOp, MetalEvalOp, MetalOpState};
use crate::tensor::MetalTensorExt;
use crate::{MetalContext, MetalTensor};
use derive_new::new;
use tract_core::internal::*;

#[derive(Clone, Debug, new, Hash)]
pub struct MetalFusedAxisOp<O: MetalEvalOp + TypedOp> {
    /// List of axis ops to apply for each op inputs
    /// Length of the list is equal to number of inputs
    pub grouped_axis_ops: TVec<TVec<MetalAxisOp>>,
    pub op: O,
}

impl<O: MetalEvalOp + TypedOp> Op for MetalFusedAxisOp<O> {
    fn name(&self) -> Cow<str> {
        self.op.name()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        let mut info = self.op.info()?;
        for (idx, axis_ops) in self.grouped_axis_ops.iter().enumerate() {
            if !axis_ops.is_empty() {
                info.push(format!(
                    "Fused axis Op on Input #{idx}: {}",
                    axis_ops
                        .iter()
                        .map(|axis_op| Ok(format!(
                            "{} - {}",
                            axis_op.name(),
                            axis_op.info()?.join(" | ")
                        )))
                        .collect::<TractResult<TVec<_>>>()?
                        .join(" | ")
                ));
            }
        }
        Ok(info)
    }

    op_as_typed_op!();
}

impl<O: MetalEvalOp + TypedOp> MetalEvalOp for MetalFusedAxisOp<O> {
    fn metal_eval(
        &self,
        context: &MetalContext,
        node_id: usize,
        session: &mut SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        // Apply Axis Ops per input
        let inputs = inputs
            .into_iter()
            .zip(self.grouped_axis_ops.iter())
            .map(|(input, axis_ops)| {
                if axis_ops.is_empty() {
                    return Ok(input);
                };
                let m_input = input.to_metal_tensor()?;
                let reshaped_input = axis_ops.iter().try_fold(
                    m_input.clone(),
                    |t, axis_op| -> TractResult<MetalTensor> {
                        let (new_shape, strides) = match &axis_op.0 {
                            AxisOp::Move(from, to) => {
                                let mut shape: TVec<usize> = t.shape().into();
                                axis_op.0.change_shape_array(&mut shape, false)?;
                                let mut strides = t.strides().to_vec();
                                strides.swap(*from, *to);
                                (shape, strides)
                            },
                            AxisOp::Reshape(at, from, to) => {
                                let strides = t.strides().to_vec();
                                ensure!(strides.is_sorted_by(|a, b| a >= b));
                                let from = from
                                    .iter()
                                    .map(|d| d.eval(&session.resolved_symbols))
                                    .collect();
                                let to =
                                    to.iter().map(|d| d.eval(&session.resolved_symbols)).collect();
                                let mut shape: TVec<usize> = t.shape().into();
                                AxisOp::Reshape(*at, from, to)
                                    .change_shape_array(&mut shape, false)?;
                                
                                (shape.clone(), Tensor::natural_strides(&shape).to_vec())
                            }
                            AxisOp::Add(ax) => {
                                let mut shape: TVec<usize> = t.shape().into();
                                axis_op.0.change_shape_array(&mut shape, false)?;

                                let mut strides = t.strides().to_vec();
                                let new_stride = strides.get((*ax as isize - 1).max(0) as usize).unwrap_or(&1);
                                strides.insert(*ax, *new_stride);
                                (shape, strides)
                            }
                            AxisOp::Rm(ax) => {
                                let mut shape: TVec<usize> = t.shape().into();
                                axis_op.0.change_shape_array(&mut shape, false)?;
                                let mut strides = t.strides().to_vec();
                                strides.remove(*ax);
                                (shape, strides)
                            }
                        };
                        t.reshaped(new_shape, strides)
                    },
                )?;

                Ok(reshaped_input.into_opaque_tensor().into())
            })
            .collect::<TractResult<TVec<_>>>()?;
        // Runner inner op
        self.op.metal_eval(context, node_id, session, inputs)
    }
}

impl<O: MetalEvalOp + TypedOp> TypedOp for MetalFusedAxisOp<O> {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(
            inputs.len() == self.grouped_axis_ops.len(),
            "Number of inputs and fused axis ops are not aligned"
        );
        // Apply AxisOp
        let inputs = inputs
            .iter()
            .zip(self.grouped_axis_ops.iter())
            .map(|(i, axis_ops)| {
                axis_ops.iter().try_fold((*i).clone(), |reshaped_i, axis_op| {
                    Ok(axis_op.output_facts(&[&reshaped_i])?[0].clone())
                })
            })
            .collect::<TractResult<TVec<_>>>()?;

        let inputs_ref = inputs.iter().collect::<TVec<_>>();
        // Apply Op
        self.op.output_facts(&inputs_ref)
    }

    as_op!();
}

impl<O: MetalEvalOp + TypedOp> EvalOp for MetalFusedAxisOp<O> {
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
