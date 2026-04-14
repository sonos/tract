use std::ops::AddAssign;

use crate::internal::*;
use crate::v2::{PulseV2Action, PulseV2Region, PulseV2Symbols, RegionTransform};
use tract_pulse_opl::tract_core::num_traits::Zero;
use tract_pulse_opl::tract_core::ops::cnn::deconv::Deconv;
use tract_pulse_opl::tract_core::ops::{FrozenOpState, OpStateFreeze};

fn deconv_transform(
    op: &dyn TypedOp,
    _source_region: &PulseV2Region,
    _symbols: &PulseV2Symbols,
) -> TractResult<Option<PulseV2Action>> {
    let deconv = op.downcast_ref::<Deconv>().unwrap();
    let dilations = deconv.pool_spec.dilations();
    let kernel_shape = &deconv.pool_spec.kernel_shape;

    // Overlap on the output = (K-1)*D for each spatial axis.
    // For 1D: just one axis.
    let axis = deconv.pool_spec.data_format.h_axis();
    let overlap = (kernel_shape[0] - 1) * dilations[0];

    if overlap == 0 {
        return Ok(None); // No overlap, default pass-through is fine.
    }

    // Wire the deconv op normally, then append PulseV2DeconvAccum.
    Ok(Some(PulseV2Action::WireOpThenPostOp(Box::new(PulseV2DeconvAccum { axis, overlap }))))
}

inventory::submit! {
    RegionTransform {
        type_id: std::any::TypeId::of::<Deconv>(),
        func: deconv_transform,
    }
}

/// Stateful op that accumulates overlapping deconv output.
///
/// The deconv produces (input_pulse + overlap) output samples per pulse.
/// This op:
/// 1. Adds the buffer (from previous pulse's tail) to the first `overlap` positions
/// 2. Saves the last `overlap` positions as the new buffer
/// 3. Emits the first (output_len - overlap) positions
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct PulseV2DeconvAccum {
    pub axis: usize,
    pub overlap: usize,
}

impl Op for PulseV2DeconvAccum {
    fn name(&self) -> StaticName {
        "PulseV2DeconvAccum".into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis={} overlap={}", self.axis, self.overlap)])
    }
    op_as_typed_op!();
}

impl EvalOp for PulseV2DeconvAccum {
    fn is_stateless(&self) -> bool {
        false
    }

    fn state(
        &self,
        _session: &TurnState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(PulseV2DeconvAccumState { buffer: None })))
    }
}

impl TypedOp for PulseV2DeconvAccum {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = inputs[0].clone();
        let dim = fact.shape[self.axis].clone() - self.overlap.to_dim();
        fact.shape.set(self.axis, dim);
        Ok(tvec!(fact))
    }
}

#[derive(Debug, Clone)]
struct PulseV2DeconvAccumState {
    buffer: Option<Tensor>,
}

impl OpStateFreeze for PulseV2DeconvAccumState {
    fn freeze(&self) -> Box<dyn FrozenOpState> {
        unimplemented!("PulseV2DeconvAccumState::freeze")
    }
}

impl OpState for PulseV2DeconvAccumState {
    fn eval(
        &mut self,
        _session: &mut TurnState,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let op = op.downcast_ref::<PulseV2DeconvAccum>().unwrap();
        let input = args_1!(inputs);
        let mut input = input.into_tensor();
        let input_len = input.shape()[op.axis];

        if input_len <= op.overlap {
            // Edge case: output is entirely overlap. Accumulate and emit nothing.
            if let Some(ref buf) = self.buffer {
                dispatch_numbers!(Self::add_buffer(input.datum_type())(
                    &mut input, buf, op.axis, input_len
                ))?;
            }
            self.buffer = Some(input);
            let mut shape = self.buffer.as_ref().unwrap().shape().to_vec();
            shape[op.axis] = 0;
            return Ok(tvec!(
                Tensor::zero_dt(self.buffer.as_ref().unwrap().datum_type(), &shape)?.into_tvalue()
            ));
        }

        // Add buffer to first `overlap` positions.
        if let Some(ref buf) = self.buffer {
            dispatch_numbers!(Self::add_buffer(input.datum_type())(
                &mut input, buf, op.axis, op.overlap
            ))?;
        }

        // Save last `overlap` positions as new buffer.
        let emit_len = input_len - op.overlap;
        self.buffer = Some(input.slice(op.axis, emit_len, input_len)?.into_tensor());

        // Emit the first part.
        let output = input.slice(op.axis, 0, emit_len)?;
        Ok(tvec!(output.into_tvalue()))
    }
}

impl PulseV2DeconvAccumState {
    fn add_buffer<T: Datum + AddAssign + Zero>(
        input: &mut Tensor,
        buffer: &Tensor,
        axis: usize,
        count: usize,
    ) -> TractResult<()> {
        let buf_len = buffer.shape()[axis];
        let add_len = count.min(buf_len);
        let mut input_view = input.to_plain_array_view_mut::<T>()?;
        let buffer_view = buffer.to_plain_array_view::<T>()?;
        let mut target = input_view.slice_axis_mut(tract_ndarray::Axis(axis), (0..add_len).into());
        let source =
            buffer_view.slice_axis(tract_ndarray::Axis(axis), (buf_len - add_len..).into());
        target += &source;
        Ok(())
    }
}
