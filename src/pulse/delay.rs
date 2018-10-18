use analyser::rules::prelude::*;
use ops::prelude::*;
use ndarray::*;

#[derive(Debug, new)]
struct DelayState {
    buffer: Tensor,
    batch: u64,
}

impl DelayState {
    pub fn eval_t<T:Datum>(&mut self, op: &Delay, input:Value) -> TfdResult<Value> {
        let axis_id = op
            .stream_input_fact
            .stream_info()?
            .ok_or("Can't delay a non streaming tensor")?
            .axis;
        let axis = Axis(axis_id);
        let input = input.to_array_view::<T>()?;
        let mut buffer = self.buffer.to_array_view_mut::<T>()?;
        if op.delay < op.pulse {
            let mut output = unsafe { ArrayD::<T>::uninitialized(input.shape()) };
            // pulse 4 delay 1 input 0 1 2 3
            // copy buffer to output beginning
            output.slice_axis_mut(axis, Slice::from(..(op.pulse-op.delay))).assign(&buffer);
            // copy 0 1 2 from input beg to output end
            output.slice_axis_mut(axis, Slice::from(op.delay..))
                .assign(&input.slice_axis(axis, Slice::from(..(op.pulse-op.delay))));
            // copy end of input to buffer
            buffer.assign(&input.slice_axis(axis, Slice::from((op.pulse-op.delay)..)));
            Ok(output.into())
        } else {
            let output = buffer.slice_axis(axis, Slice::from(0..op.pulse)).to_owned();
            let stride = buffer.strides()[axis_id] as usize * op.pulse;
            buffer.as_slice_mut().unwrap().rotate_left(stride);
            buffer.slice_axis_mut(axis, Slice::from((op.delay - op.pulse)..)).assign(&input);
            Ok(output.into())
        }
    }
}

impl OpState for DelayState {
    fn eval(&mut self, op: &Op, mut inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        let input = args_1!(inputs);
        let op = op.downcast_ref::<Delay>().ok_or("Wrong Op type")?;
        Ok(tvec!(dispatch_datum!(Self::eval_t(input.datum_type())(self, op, input))?))
    }
}

#[derive(Clone, Default, Debug, new)]
struct Delay {
    stream_input_fact: TensorFact,
    pulse: usize,
    delay: usize,
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
        let dt = self
            .stream_input_fact
            .datum_type
            .concretize()
            .ok_or("Delay with abstract tensor type")?;
        let shape = self
            .stream_input_fact
            .shape
            .concretize()
            .ok_or("Delay with abstract tensor shape")?;
        let axis = self
            .stream_input_fact
            .stream_info()?
            .ok_or("Can't delay a non streaming tensor")?
            .axis;
        let buffer_shape: Vec<_> = shape
            .iter()
            .enumerate()
            .map(|(ax, &dim)| {
                if ax != axis {
                    dim.to_integer().unwrap() as usize
                } else {
                    self.delay
                }
            }).collect();
        let buffer = dispatch_datum!(self::make_buffer(dt)(&buffer_shape));
        Ok(Some(Box::new(DelayState { buffer, batch: 0 })))
    }
}

impl InferenceRulesOp for Delay {
    /// Registers the inference rules of the operator.
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 0)?;
        s.equals(&outputs.len, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        let axis = self
            .stream_input_fact
            .stream_info()?
            .ok_or("Can't delay a non streaming tensor")?
            .axis;
        s.given(&inputs[0].rank, move |s, rank| {
            for ax in 0..(rank as usize) {
                if ax == axis {
                    s.equals(&outputs[0].shape[ax], self.pulse.to_dim())?;
                } else {
                    s.equals(&outputs[0].shape[ax], &inputs[0].shape[ax])?;
                }
            }
            Ok(())
        })?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use ::*;
    use super::*;
    use model::dsl::*;

    fn test_pulse_delay(pulse: usize, delay: usize) {
        let mut model = Model::default();
        let stream_fact = TensorFact::dt_shape(u8::datum_type(), vec!(TDim::s()));
        let pulse_fact = TensorFact::dt_shape(u8::datum_type(), vec!(pulse));
        model.add_source_fact("source", pulse_fact).unwrap();
        model.chain("delay", Box::new(Delay::new(stream_fact, pulse, delay))).unwrap();

        let plan = SimplePlan::new(model).unwrap();
        let mut state = ::plan::SimpleState::new(plan).unwrap();

        for i in 0..5 {
            let input = arr1(&[0u8, 1, 2, 3]) + 4 * i;
            let expect = input.mapv(|i| i.saturating_sub(delay as u8) as u8);
            state.reset().unwrap();
            state.set_input(0, input.into()).unwrap();
            state.eval_all_in_order().unwrap();
            let output = state.take_outputs().unwrap();
            assert_eq!(output[0].as_u8s().unwrap(), &expect.into_dimensionality().unwrap());
        }
    }

    #[test]
    fn sub_pulse() {
        test_pulse_delay(4, 1);
    }

    #[test]
    fn supra_pulse() {
        test_pulse_delay(4, 5);
    }
}
