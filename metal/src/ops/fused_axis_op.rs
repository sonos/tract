use crate::ops::MetalAxisOp;
use derive_new::new;
use tract_core::internal::tract_smallvec::ToSmallVec;
use tract_core::internal::*;
use tract_core::ops::OpStateFreeze;
use tract_gpu::tensor::{DeviceTensor, DeviceTensorExt};

#[derive(Clone, Debug, new)]
pub struct MetalFusedAxisOp {
    /// List of axis ops to apply for each op inputs
    /// Length of the list is equal to number of inputs
    pub grouped_axis_ops: TVec<TVec<MetalAxisOp>>,
    pub op: Box<dyn TypedOp>,
}

#[derive(Debug, Clone, new)]
pub struct MetalFusedAxisOpState {
    pub op_state: Box<dyn OpState>,
}

fn compute_reshaped_inputs(
    inputs: TVec<TValue>,
    grouped_axis_ops: &TVec<TVec<MetalAxisOp>>,
    session: &SessionState,
) -> TractResult<TVec<TValue>> {
    // Apply Axis Ops per input

    inputs
        .into_iter()
        .zip(grouped_axis_ops.iter())
        .map(|(input, axis_ops)| {
            if axis_ops.is_empty() {
                return Ok(input);
            };
            let m_input = input.to_device_tensor()?;
            let reshaped_input = axis_ops.iter().try_fold(
                m_input.clone(),
                |t, axis_op| -> TractResult<DeviceTensor> {
                    let new_shape = match &axis_op.0 {
                        AxisOp::Reshape(skip, from, to) => {
                            let from =
                                from.iter().map(|d| d.eval(&session.resolved_symbols)).collect();
                            let to = to.iter().map(|d| d.eval(&session.resolved_symbols)).collect();
                            let mut shape: TVec<usize> = t.shape().into();
                            AxisOp::Reshape(*skip, from, to)
                                .change_shape_array(&mut shape, false)?;
                            shape
                        }
                        AxisOp::Add(_) | AxisOp::Rm(_) | AxisOp::Move(..) => {
                            let mut shape: TVec<usize> = t.shape().into();
                            axis_op.0.change_shape_array(&mut shape, false)?;
                            shape
                        }
                    };
                    if let AxisOp::Move(from, to) = axis_op.0 {
                        let mut out_strides: TVec<isize> = t.strides().to_smallvec();
                        let removed_stride = out_strides.remove(from);
                        out_strides.insert(to, removed_stride);
                        let tmp_t = t.reshaped(new_shape)?;
                        tmp_t.restrided(out_strides)
                    } else {
                        t.reshaped(new_shape)
                    }
                },
            )?;

            Ok(reshaped_input.into_opaque_tensor().into())
        })
        .collect::<TractResult<TVec<_>>>()
}

impl OpState for MetalFusedAxisOpState {
    fn init_tensor_fact(&self) -> Option<(String, TypedFact)> {
        self.op_state.init_tensor_fact()
    }

    fn load_from(
        &mut self,
        session: &mut SessionState,
        states: &mut Vec<TValue>,
    ) -> TractResult<()> {
        self.op_state.load_from(session, states)
    }

    fn save_to(&self, states: &mut Vec<TValue>) -> TractResult<()> {
        self.op_state.save_to(states)
    }

    fn resolve_symbols(&mut self, session: &mut SessionState) -> TractResult<()> {
        self.op_state.resolve_symbols(session)
    }

    fn eval(
        &mut self,
        session: &mut SessionState,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let fused_axis_op = op.downcast_ref::<MetalFusedAxisOp>().unwrap();
        let inputs = compute_reshaped_inputs(inputs, &fused_axis_op.grouped_axis_ops, session)?;
        // Runner inner op
        self.op_state.eval(session, fused_axis_op.op.as_op(), inputs)
    }
}

#[derive(Debug, Clone)]
pub struct FrozenMetalFusedAxisOpState {
    pub op_state: Box<dyn FrozenOpState>,
}

impl OpStateFreeze for MetalFusedAxisOpState {
    fn freeze(&self) -> Box<dyn FrozenOpState + 'static> {
        Box::new(FrozenMetalFusedAxisOpState { op_state: self.op_state.freeze() })
    }
}

impl FrozenOpState for FrozenMetalFusedAxisOpState {
    fn unfreeze(&self) -> Box<dyn OpState> {
        Box::new(MetalFusedAxisOpState { op_state: self.op_state.unfreeze() })
    }
}

impl Op for MetalFusedAxisOp {
    fn name(&self) -> StaticName {
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

impl EvalOp for MetalFusedAxisOp {
    fn is_stateless(&self) -> bool {
        self.op.is_stateless()
    }

    fn state(
        &self,
        session: &mut SessionState,
        node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        if let Some(state) = self.op.state(session, node_id)? {
            Ok(Some(Box::new(MetalFusedAxisOpState { op_state: state })))
        } else {
            Ok(None)
        }
    }
    fn eval_with_session(
        &self,
        node_id: usize,
        session: &SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let inputs = compute_reshaped_inputs(inputs, &self.grouped_axis_ops, session)?;
        // Runner inner op
        self.op.eval_with_session(node_id, session, inputs)
    }
}

impl TypedOp for MetalFusedAxisOp {
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
