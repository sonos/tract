use std::ops::AddAssign;

use tract_ndarray::Axis;
use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::OpStateFreeze;
use tract_num_traits::Zero;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeconvDelay {
    pub axis: usize,
    pub overlap: usize,
    pub delay: usize,
    pub stride: usize,
    pub pulse: TDim,
    pub deconv_input_dim: TDim,
    pub deconv_output_dim: TDim,
}



impl Op for DeconvDelay {
    fn name(&self) -> Cow<str> {
        "DeconvDelay".into()
    }

    op_as_typed_op!();
}

impl EvalOp for DeconvDelay {
    fn is_stateless(&self) -> bool {
        false
    }

    fn eval(&self, _inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        unreachable!()
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(DeconvDelayState { valid_inputed: -(self.delay as isize), buffer: None })))
    }
}

impl TypedOp for DeconvDelay {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = inputs[0].clone();
        let len = fact.shape[self.axis].clone();
        fact.shape.set(self.axis, len - self.overlap);
        Ok(tvec!(fact))
    }

    as_op!();
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct DeconvDelayState {
    valid_inputed: isize,
    buffer: Option<Tensor>,
}

impl OpState for DeconvDelayState {
    fn eval(
        &mut self,
        session: &mut SessionState,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let op = op.downcast_ref::<DeconvDelay>().context("Wrong op")?;
        if self.buffer.is_none() {
            let mut buffer_size: TVec<usize> = inputs[0].shape().into();
            buffer_size[op.axis] = op.overlap; //+ (op.stride - 1) * (op.pulse - 1);
            self.buffer = Some(Tensor::zero_dt(inputs[0].datum_type(), &buffer_size)?);
        }
        let mut input = inputs[0].clone().into_tensor();
        dispatch_numbers!(Self::eval_t(input.datum_type())(self, session, op, &mut input))?;
        let output = input.slice(op.axis, 0, input.shape()[op.axis] - op.overlap)?;
        Ok(tvec!(output.into_tvalue()))
    }
}

impl DeconvDelayState {
    fn eval_t<T: Datum + AddAssign + Zero>(
        &mut self,
        session: &SessionState,
        op: &DeconvDelay,
        input: &mut Tensor,
    ) -> TractResult<()> {
        let buffer = self.buffer.as_mut().unwrap();
        let mut buffer = buffer.to_array_view_mut::<T>()?;
        let mut input = input.to_array_view_mut::<T>()?;
        let input_pulse = input.shape()[op.axis];
        let output_pulse = input_pulse - op.overlap;
        self.valid_inputed += output_pulse as isize;
        if let Ok(input_dim) = op.deconv_input_dim.eval(&session.resolved_symbols).to_isize() {
            if self.valid_inputed > input_dim {
                let to_be_zeroed = ((self.valid_inputed - input_dim) as usize).min(input_pulse);
                let mut zeroed =
                    input.slice_axis_mut(Axis(op.axis), (input_pulse - to_be_zeroed..).into());
                zeroed.fill(T::zero());
            }
        }
        {
            let mut input_view = input.slice_axis_mut(Axis(op.axis), (0..op.overlap).into());
            input_view += &buffer;
        }
        buffer.assign(&input.slice_axis(Axis(op.axis), (output_pulse..).into()));

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
struct FrozenDeconvDelayState {
    valid_inputed: isize,
    buffer: Option<Arc<Tensor>>,
}

impl OpStateFreeze for DeconvDelayState {
    fn freeze(&self) -> Box<dyn FrozenOpState> {
        Box::new(FrozenDeconvDelayState {
            valid_inputed: self.valid_inputed,
            buffer: self.buffer.as_ref().map(|t| t.clone().into_arc_tensor()),
        })
    }
}

impl FrozenOpState for FrozenDeconvDelayState {
    fn unfreeze(&self) -> Box<dyn OpState> {
        Box::new(DeconvDelayState {
            valid_inputed: self.valid_inputed,
            buffer: self.buffer.as_ref().map(|t| t.clone().into_tensor()),
        })
    }
}
