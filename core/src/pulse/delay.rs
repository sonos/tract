use crate::internal::*;
use crate::pulse::PulsedTensorFact;
use ndarray::*;

#[derive(Debug, new, Clone)]
struct DelayState {
    buffer: Tensor,
}

impl DelayState {
    pub fn eval_t<T: Datum>(
        &mut self,
        op: &Delay,
        input: Arc<Tensor>,
    ) -> TractResult<Arc<Tensor>> {
        let axis = Axis(op.input_fact.axis);
        let input = input.to_array_view::<T>()?;
        let mut buffer = self.buffer.to_array_view_mut::<T>()?;

        let buffered = op.delay + op.overlap;
        let mut output_shape: TVec<_> = op.input_fact.shape.clone();
        let input_pulse = op.input_fact.pulse();
        let output_pulse = input_pulse + op.overlap;
        output_shape[op.input_fact.axis] = output_pulse;
        // build output
        let output = if op.delay < input_pulse {
            let mut output = unsafe { T::uninitialized_array(&*output_shape) };
            let from_input = input_pulse - op.delay;
            let from_buffer = output_pulse - from_input;
            output
                .slice_axis_mut(axis, Slice::from(..from_buffer))
                .assign(&buffer.slice_axis(axis, Slice::from(..from_buffer)));
            output
                .slice_axis_mut(axis, Slice::from(from_buffer..))
                .assign(&input.slice_axis(axis, Slice::from(..from_input)));
            output
        } else {
            buffer.slice_axis(axis, Slice::from(..output_pulse)).to_owned()
        };
        // maintain buffer
        if buffered < input_pulse {
            buffer.assign(&input.slice_axis(axis, Slice::from((input_pulse - buffered)..)));
        } else {
            let stride = buffer.strides()[op.input_fact.axis] as usize * input_pulse;
            buffer.as_slice_mut().unwrap().rotate_left(stride);
            buffer.slice_axis_mut(axis, Slice::from((buffered - input_pulse)..)).assign(&input);
        }
        Ok(output.into_arc_tensor())
    }
}

impl OpState for DelayState {
    fn eval(
        &mut self,
        _state: &mut SessionState,
        op: &dyn Op,
        mut inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let op = op.downcast_ref::<Delay>().ok_or("Wrong Op type")?;
        Ok(tvec!(dispatch_datum!(Self::eval_t(input.datum_type())(self, op, input))?))
    }
}

#[derive(Clone, Debug, new, PartialEq)]
pub struct Delay {
    input_fact: PulsedTensorFact,
    delay: usize,
    overlap: usize,
}

impl Op for Delay {
    fn name(&self) -> Cow<str> {
        "Delay".into()
    }

    impl_op_same_as!();
    to_typed!();
}

fn make_buffer<T: Datum>(shape: &[usize]) -> Tensor {
    ::ndarray::ArrayD::<T>::default(shape).into()
}

impl StatefullOp for Delay {
    fn state(&self, _session: &mut SessionState, _node_id: usize) -> TractResult<Option<Box<dyn OpState>>> {
        let mut buffer_shape: TVec<_> = self.input_fact.shape.clone();
        buffer_shape[self.input_fact.axis] = self.delay + self.overlap;
        let buffer = dispatch_datum!(self::make_buffer(self.input_fact.dt)(&buffer_shape));
        Ok(Some(Box::new(DelayState { buffer })))
    }
}

impl TypedOp for Delay {
    typed_op_as_op!();
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::*;

    fn test_pulse_delay_over(pulse: usize, delay: usize, overlap: usize) {
        let mut model = PulsedModel::default();
        let fact1 = PulsedTensorFact {
            dt: u8::datum_type(),
            shape: tvec![pulse],
            axis: 0,
            dim: TDim::s(),
            delay: 0,
        };
        model.add_source("source", fact1.clone()).unwrap();
        let fact2 = PulsedTensorFact {
            dt: u8::datum_type(),
            shape: tvec![pulse + overlap],
            axis: 0,
            dim: TDim::s(),
            delay,
        };
        model.chain("delay", Delay::new(fact1, delay, overlap), tvec!(fact2)).unwrap();
        model.auto_outputs().unwrap();

        let plan = SimplePlan::new(model).unwrap();
        let mut state = crate::plan::SimpleState::new(plan).unwrap();

        for i in 0..5 {
            let input: Vec<u8> = (pulse * i..(pulse * (i + 1))).map(|a| a as u8).collect();
            let expect: Vec<u8> = (pulse * i..(pulse * (i + 1) + overlap))
                .map(|i| i.saturating_sub(delay + overlap) as u8)
                .collect();
            let output = state.run(tvec!(Tensor::from(arr1(&input)))).unwrap();
            assert_eq!(output[0].to_array_view::<u8>().unwrap().as_slice().unwrap(), &*expect);
        }
    }

    #[test]
    fn sub_pulse() {
        test_pulse_delay_over(4, 1, 0);
    }

    #[test]
    fn supra_pulse() {
        test_pulse_delay_over(4, 5, 0);
    }

    #[test]
    fn sub_pulse_context() {
        test_pulse_delay_over(4, 0, 2);
    }

    #[test]
    fn supra_pulse_context() {
        test_pulse_delay_over(4, 0, 6);
    }

    #[test]
    fn test_two_delays() {
        let pulse = 4;
        let mut model = PulsedModel::default();
        let fact_0 = PulsedTensorFact {
            dt: u8::datum_type(),
            shape: tvec![pulse],
            axis: 0,
            dim: TDim::s(),
            delay: 0,
        };
        model.add_source("source", fact_0.clone()).unwrap();
        let fact_1 = PulsedTensorFact {
            dt: u8::datum_type(),
            shape: tvec![pulse],
            axis: 0,
            dim: TDim::s(),
            delay: 2,
        };
        model.chain("delay-1", Delay::new(fact_0, 2, 0), tvec!(fact_1.clone())).unwrap();
        let fact_2 = PulsedTensorFact {
            dt: u8::datum_type(),
            shape: tvec![pulse],
            axis: 0,
            dim: TDim::s(),
            delay: 4,
        };
        model.chain("delay-2", Delay::new(fact_1, 2, 0), tvec!(fact_2)).unwrap();
        model.auto_outputs().unwrap();

        let plan = SimplePlan::new(model).unwrap();
        let mut state = crate::plan::SimpleState::new(plan).unwrap();

        for i in 0..5 {
            let input: Vec<u8> = (pulse * i..(pulse * (i + 1))).map(|a| a as u8).collect();
            let expect: Vec<u8> =
                (pulse * i..(pulse * (i + 1))).map(|i| i.saturating_sub(4) as u8).collect();
            let output = state.run(tvec!(Tensor::from(arr1(&input)))).unwrap();
            assert_eq!(output[0].to_array_view::<u8>().unwrap().as_slice().unwrap(), &*expect);
        }
    }
}
