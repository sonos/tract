use tract_nnef::internal::*;

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_pulse_delay",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Integer.named("axis"),
            TypeName::Integer.named("delay"),
            TypeName::Integer.named("overlap"),
        ],
        de_delay,
    );
}

fn de_delay(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let wire = invocation.named_arg_as(builder, "input")?;
    let axis = invocation.named_arg_as::<i64>(builder, "axis")? as usize;
    let delay = invocation.named_arg_as::<i64>(builder, "delay")? as usize;
    let overlap = invocation.named_arg_as::<i64>(builder, "overlap")? as usize;
    let input_fact = builder.model.outlet_fact(wire)?;
    let op = Delay::new_typed(input_fact, axis, delay, overlap)?;
    builder.wire(op, &[wire])
}

#[derive(Debug, Clone)]
struct DelayState {
    buffer: Tensor,
}

impl OpState for DelayState {
    fn eval(
        &mut self,
        _state: &mut SessionState,
        op: &dyn Op,
        mut inputs: TVec<TensorVar>,
    ) -> TractResult<TVec<Box<Tensor>>> {
        let input = args_1!(inputs);
        let op = op.downcast_ref::<Delay>().ok_or_else(|| format_err!("Wrong Op type"))?;
        let buffered = op.delay + op.overlap;
        let input_pulse = input.shape()[op.axis];
        let output_pulse = input_pulse + op.overlap;
        let mut output_shape: TVec<usize> = input.shape().into();
        output_shape[op.axis] = output_pulse;
        // build output
        unsafe {
            if self.buffer.rank() == 0 {
                let mut shape = input.shape().to_owned();
                shape[op.axis] = buffered;
                self.buffer = Tensor::uninitialized_dt(input.datum_type(), &shape)?;
            }
            let mut output = Tensor::uninitialized_dt(input.datum_type(), &*output_shape)?;
            if op.delay < input_pulse {
                let from_input = input_pulse - op.delay;
                let from_buffer = output_pulse - from_input;
                output.assign_slice_unchecked(..from_buffer, &self.buffer, ..from_buffer, op.axis);
                output.assign_slice_unchecked(from_buffer.., &input, ..from_input, op.axis);
            } else {
                output.assign_slice_unchecked(.., &self.buffer, ..output_pulse, op.axis);
            };
            // maintain buffer
            if buffered < input_pulse {
                self.buffer.assign_slice_unchecked(.., &input, (input_pulse - buffered).., op.axis);
            } else {
                let stride = self.buffer.shape().iter().skip(op.axis + 1).product::<usize>()
                    * input.datum_type().size_of()
                    * input_pulse;
                std::slice::from_raw_parts_mut(
                    self.buffer.as_ptr_mut_unchecked::<u8>(),
                    self.buffer.len() * input.datum_type().size_of(),
                )
                .rotate_left(stride);
                self.buffer.assign_slice_unchecked((buffered - input_pulse).., &input, .., op.axis);
            }
            Ok(tvec!(output.boxed()))
        }
    }
}

#[derive(Clone, Debug, PartialEq, Hash)]
pub struct Delay {
    pub datum_type: DatumType,
    pub buffer_shape: TVec<TDim>,
    pub axis: usize,
    pub delay: usize,
    pub overlap: usize,
}

impl_dyn_hash!(Delay);

impl Delay {
    pub fn new(axis: usize, input_fact: &TypedFact, delay: usize, overlap: usize) -> Delay {
        let axis = axis;
        let mut buffer_shape = input_fact.shape.to_tvec();
        buffer_shape[axis] = (delay + overlap).to_dim();
        Delay { datum_type: input_fact.datum_type, buffer_shape, axis, delay, overlap }
    }

    pub fn new_typed(
        input_fact: &TypedFact,
        axis: usize,
        delay: usize,
        overlap: usize,
    ) -> TractResult<Delay> {
        let mut buffer_shape: TVec<TDim> = input_fact.shape.iter().map(|d| d.clone()).collect();
        buffer_shape[axis] = (delay + overlap).to_dim();
        Ok(Delay { datum_type: input_fact.datum_type, buffer_shape, axis, delay, overlap })
    }
}

impl Op for Delay {
    fn name(&self) -> Cow<str> {
        "Delay".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![
            format!("axis: {} delay: {} overlap: {}", self.axis, self.delay, self.overlap),
            format!("buffer: {:?} {:?}", self.buffer_shape, self.datum_type),
        ])
    }

    op_pulse!();
    impl_op_same_as!();
    op_as_typed_op!();
}

impl EvalOp for Delay {
    fn is_stateless(&self) -> bool {
        false
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(DelayState { buffer: tensor0(0.0f32) })))
    }
}

impl TypedOp for Delay {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = inputs[0].clone();
        fact.shape.set(self.axis, fact.shape[self.axis].clone() + self.overlap.to_dim());
        Ok(tvec!(fact))
    }

    fn cost(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        Ok(tvec!((Cost::Buffer(self.datum_type), self.buffer_shape.iter().maybe_product()?)))
    }
}
