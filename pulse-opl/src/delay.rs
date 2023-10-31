use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::OpStateFreeze;

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_pulse_delay",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Integer.named("axis"),
            TypeName::Integer.named("delay"),
            TypeName::Integer.named("overlap"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_delay,
    );
}

fn de_delay(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let wire = invocation.named_arg_as(builder, "input")?;
    let axis = invocation.named_arg_as::<i64>(builder, "axis")? as usize;
    let delay = invocation.named_arg_as::<i64>(builder, "delay")? as usize;
    let overlap = invocation.named_arg_as::<i64>(builder, "overlap")? as usize;
    let input_fact = builder.model.outlet_fact(wire)?;
    let op = Delay::new_typed(input_fact, axis, delay, overlap);
    builder.wire(op, &[wire])
}

#[derive(Debug, Clone)]
pub struct DelayState {
    pub buffer: Option<Tensor>,
}

impl DelayState {
    /// Apply delay op on input and store the result in the output tensor
    /// This method doesn't use allocation.
    ///
    /// # Safety
    ///
    /// Input and Ouput tensors shape must be compatible with this operator, otherwise it could lead
    /// to an undefined behaviour.
    pub unsafe fn apply_delay_unchecked(
        &mut self,
        op: &Delay,
        input: &Tensor,
        output: &mut Tensor,
    ) {
        let buffered = op.delay + op.overlap;
        let input_pulse = input.shape()[op.axis];
        let output_pulse = input_pulse + op.overlap;
        let buffer = self.buffer.as_mut().unwrap();
        if op.delay < input_pulse {
            let from_input = input_pulse - op.delay;
            let from_buffer = output_pulse - from_input;
            output.assign_slice_unchecked(..from_buffer, buffer, ..from_buffer, op.axis);
            output.assign_slice_unchecked(from_buffer.., input, ..from_input, op.axis);
        } else {
            output.assign_slice_unchecked(.., buffer, ..output_pulse, op.axis);
        };
        // maintain buffer
        if buffered < input_pulse {
            buffer.assign_slice_unchecked(.., input, (input_pulse - buffered).., op.axis);
        } else {
            let stride = buffer.shape().iter().skip(op.axis + 1).product::<usize>()
                * input.datum_type().size_of()
                * input_pulse;
            std::slice::from_raw_parts_mut(
                buffer.as_ptr_mut_unchecked::<u8>(),
                buffer.len() * input.datum_type().size_of(),
            )
            .rotate_left(stride);
            buffer.assign_slice_unchecked((buffered - input_pulse).., input, .., op.axis);
        }
    }
}

impl OpState for DelayState {
    fn eval(
        &mut self,
        _state: &mut SessionState,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let op = op.downcast_ref::<Delay>().ok_or_else(|| format_err!("Wrong Op type"))?;
        let buffered = op.delay + op.overlap;
        let input_pulse = input.shape()[op.axis];
        let output_pulse = input_pulse + op.overlap;
        let mut output_shape: TVec<usize> = input.shape().into();
        output_shape[op.axis] = output_pulse;
        // build output
        unsafe {
            if self.buffer.is_none() {
                let mut shape = input.shape().to_owned();
                shape[op.axis] = buffered;
                self.buffer = Some(Tensor::uninitialized_dt(input.datum_type(), &shape)?);
            };
            let mut output = Tensor::uninitialized_dt(input.datum_type(), &output_shape)?;
            self.apply_delay_unchecked(op, &input, &mut output);
            Ok(tvec!(output.into()))
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Delay {
    pub buffer_shape: TVec<TDim>,
    pub axis: usize,
    pub delay: usize,
    pub overlap: usize,
}

impl Delay {
    pub fn new_typed(input_fact: &TypedFact, axis: usize, delay: usize, overlap: usize) -> Delay {
        let mut buffer_shape: TVec<TDim> = input_fact.shape.to_tvec();
        buffer_shape[axis] = (delay + overlap).to_dim();
        Delay { buffer_shape, axis, delay, overlap }
    }
}

impl Op for Delay {
    fn name(&self) -> Cow<str> {
        "Delay".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![
            format!("axis: {} delay: {} overlap: {}", self.axis, self.delay, self.overlap),
            format!("buffer: {:?}", self.buffer_shape),
        ])
    }

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
        Ok(Some(Box::new(DelayState { buffer: None })))
    }
}

impl TypedOp for Delay {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = inputs[0].clone();
        fact.shape.set(self.axis, fact.shape[self.axis].clone() + self.overlap.to_dim());
        Ok(tvec!(fact))
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        Ok(tvec!((Cost::Buffer(inputs[0].datum_type), self.buffer_shape.iter().product())))
    }

    fn suggested_axis_changes(&self) -> TractResult<TVec<(InOut, AxisOp)>> {
        if self.axis != 0 {
            Ok(tvec!((InOut::In(0), AxisOp::Move(self.axis, 0))))
        } else {
            Ok(tvec!())
        }
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        if let Some(axis) = change.transform_axis(self.axis) {
            if axis != self.axis {
                Ok(Some(AxisChangeConsequence::new(
                    model,
                    node,
                    Some(Box::new(Self { axis, ..self.clone() }) as _),
                    change,
                )))
            } else {
                Ok(Some(AxisChangeConsequence::new(model, node, None, change)))
            }
        } else {
            Ok(None)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
struct FrozenDelayState {
    buffer: Option<Arc<Tensor>>,
}

impl OpStateFreeze for DelayState {
    fn freeze(&self) -> Box<dyn FrozenOpState> {
        Box::new(FrozenDelayState { buffer: self.buffer.as_ref().map(|t| t.clone().into_arc_tensor()) })
    }
}

impl FrozenOpState for FrozenDelayState {
    fn unfreeze(&self) -> Box<dyn OpState> {
        Box::new(DelayState { buffer: self.buffer.as_ref().map(|t| t.clone().into_tensor()) })
    }
}
