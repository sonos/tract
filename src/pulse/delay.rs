use ndarray::*;
use ops::prelude::*;

#[derive(Debug, new)]
struct DelayState {
    buffer: Tensor,
    batch: u64,
}

impl DelayState {
    pub fn eval_t<T: Datum>(&mut self, op: &Delay, input: Value) -> TfdResult<Value> {
        let axis = Axis(op.input_fact.axis);
        let input = input.to_array_view::<T>()?;
        let mut buffer = self.buffer.to_array_view_mut::<T>()?;

        let buffered = op.delay + op.overlap;
        let mut output_shape: Vec<_> = op.input_fact.shape.clone();
        let input_pulse = op.input_fact.pulse();
        let output_pulse = input_pulse + op.overlap;
        output_shape[op.input_fact.axis] = output_pulse;
        // build output
        let output = if op.delay < input_pulse  {
            let mut output = unsafe { ArrayD::<T>::uninitialized(output_shape) };
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
            buffer
                .slice_axis_mut(axis, Slice::from((buffered - input_pulse)..))
                .assign(&input);
        }
        Ok(output.into())
    }
}

impl OpState for DelayState {
    fn eval(&mut self, op: &Op, mut inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        let input = args_1!(inputs);
        let op = op.downcast_ref::<Delay>().ok_or("Wrong Op type")?;
        Ok(tvec!(dispatch_datum!(Self::eval_t(input.datum_type())(
            self, op, input
        ))?))
    }
}

#[derive(Clone, Debug, new)]
pub struct Delay {
    input_fact: PulsedTensorFact,
    delay: usize,
    overlap: usize,
}

impl Op for Delay {
    fn name(&self) -> &str {
        "Delay"
    }
}

fn make_buffer<T: Datum>(shape: &[usize]) -> Tensor {
    ::ndarray::ArrayD::<T>::default(shape).into()
}

impl StatefullOp for Delay {
    fn state(&self) -> TfdResult<Option<Box<OpState>>> {
        let mut buffer_shape: Vec<_> = self.input_fact.shape.clone();
        buffer_shape[self.input_fact.axis] = self.delay + self.overlap;
        let buffer = dispatch_datum!(self::make_buffer(self.input_fact.dt)(&buffer_shape));
        Ok(Some(Box::new(DelayState { buffer, batch: 0 })))
    }
}

impl InferenceRulesOp for Delay {
    /// Registers the inference rules of the operator.
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        _s: &mut Solver<'r>,
        _inputs: &'p TensorsProxy,
        _outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use model::dsl::*;
    use *;

    fn test_pulse_delay_over(pulse: usize, delay: usize, overlap: usize) {
        let mut model = Model::default();
        let fact = PulsedTensorFact {
            dt: u8::datum_type(),
            shape: vec!(pulse),
            axis: 0,
            dim: TDim::s(),
            delay: 0,
        };
        model.add_source_fact("source", fact.to_pulse_fact()).unwrap();
        model
            .chain(
                "delay",
                Box::new(Delay::new(fact, delay, overlap)),
            ).unwrap();

        let plan = SimplePlan::new(model).unwrap();
        let mut state = ::plan::SimpleState::new(plan).unwrap();

        for i in 0..5 {
            let input: Vec<u8> = (pulse * i..(pulse * (i + 1))).map(|a| a as u8).collect();
            let expect: Vec<u8> = (pulse * i..(pulse * (i + 1) + overlap))
                .map(|i| i.saturating_sub(delay + overlap) as u8)
                .collect();
            state.reset().unwrap();
            state.set_input(0, Tensor::from(arr1(&input))).unwrap();
            state.eval_all_in_order().unwrap();
            let output = state.take_outputs().unwrap();
            assert_eq!(
                output[0].as_u8s().unwrap().as_slice().unwrap(),
                &*expect
            );
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
        let mut model = Model::default();
        let fact = PulsedTensorFact {
            dt: u8::datum_type(),
            shape: vec!(pulse),
            axis: 0,
            dim: TDim::s(),
            delay: 0,
        };
        model.add_source_fact("source", fact.to_pulse_fact()).unwrap();
        model
            .chain(
                "delay-1",
                Box::new(Delay::new(fact, 2, 0)),
            ).unwrap();
        let fact = PulsedTensorFact {
            dt: u8::datum_type(),
            shape: vec!(pulse),
            axis: 0,
            dim: TDim::s(),
            delay: 2,
        };
        model
            .chain(
                "delay-2",
                Box::new(Delay::new(fact, 2, 0))
            ).unwrap();

        let plan = SimplePlan::new(model).unwrap();
        let mut state = ::plan::SimpleState::new(plan).unwrap();

        for i in 0..5 {
            let input: Vec<u8> = (pulse * i..(pulse * (i + 1))).map(|a| a as u8).collect();
            let expect: Vec<u8> = (pulse * i..(pulse * (i + 1)))
                .map(|i| i.saturating_sub(4) as u8)
                .collect();
            state.reset().unwrap();
            state.set_input(0, Tensor::from(arr1(&input))).unwrap();
            state.eval_all_in_order().unwrap();
            let output = state.take_outputs().unwrap();
            assert_eq!(
                output[0].as_u8s().unwrap().as_slice().unwrap(),
                &*expect
            );
        }
    }
}
