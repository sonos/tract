use tract_nnef::internal::*;
use tract_nnef::ser::tdim;
use tract_nnef::tract_core::trivial_op_state_freeeze;

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_pulse_mask",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Integer.named("axis"),
            TypeName::Integer.named("begin"),
            TypeName::Integer.named("end"),
            TypeName::Scalar.named("value"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        deser,
    );
    registry.register_dumper(ser)
}

fn ser(ast: &mut IntoAst, node: &TypedNode, op: &PulseMask) -> TractResult<Option<Arc<RValue>>> {
    let wire = ast.mapping[&node.inputs[0]].clone();
    let params = vec![
        ("axis", numeric(op.axis)),
        ("begin", numeric(op.begin)),
        ("end", tdim(&op.end)),
        ("value", numeric(op.value.cast_to_scalar::<f32>())),
    ];
    Ok(Some(invocation("tract_pulse_mask", &[wire], &params)))
}

fn deser(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let wire = invocation.named_arg_as(builder, "input")?;
    let axis = invocation.named_arg_as(builder, "axis")?;
    let begin = invocation.named_arg_as(builder, "begin")?;
    let value: Tensor = tensor0(invocation.named_arg_as::<f32>(builder, "value")?);
    let end = builder.allowing_new_symbols(|builder| invocation.named_arg_as(builder, "end"))?;
    let op = PulseMask { axis, begin, end, value };
    builder.wire(op, &[wire])
}

#[derive(Debug, Clone, Default, Hash)]
struct PulseMaskOpState {
    current_pos: usize,
}

impl OpState for PulseMaskOpState {
    fn eval(
        &mut self,
        session: &mut SessionState,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs).into_tensor();
        let op = op.downcast_ref::<PulseMask>().ok_or_else(|| format_err!("Wrong Op type"))?;
        let tensor = self.pad(session, op, input)?;
        Ok(tvec!(tensor.into_tvalue()))
    }
}

impl PulseMaskOpState {
    fn pad(
        &mut self,
        session: &SessionState,
        op: &PulseMask,
        mut input: Tensor,
    ) -> TractResult<Tensor> {
        let pulse = input.shape()[op.axis];
        let pulse_begin = self.current_pos;
        let pulse_end = self.current_pos + pulse;
        self.current_pos += pulse;
        let end = op.end.eval(&session.resolved_symbols).to_usize().unwrap_or(usize::MAX);

        // pulse is entirely in valid input, just forward
        if pulse_begin >= op.begin && pulse_end <= end {
            return Ok(input);
        }

        if pulse_begin < op.begin {
            let fill_up_to = (op.begin - pulse_begin).min(pulse);
            unsafe {
                dispatch_copy_by_size!(crate::pad::fill_slice_constant(input.datum_type())(
                    &mut input,
                    &op.value,
                    op.axis,
                    0..fill_up_to
                ))
            };
        }
        if pulse_end > end {
            let fill_from = pulse - (pulse_end - end).min(pulse);
            unsafe {
                dispatch_copy_by_size!(crate::pad::fill_slice_constant(input.datum_type())(
                    &mut input,
                    &op.value,
                    op.axis,
                    fill_from..pulse
                ))
            }
        }

        Ok(input)
    }
}

#[derive(Debug, Clone, Default, Hash)]
pub struct PulseMask {
    pub axis: usize,
    pub begin: usize,
    pub end: TDim,
    pub value: Tensor,
}

impl Op for PulseMask {
    fn name(&self) -> Cow<str> {
        "PulseMask".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {} begin: {} end: {}", self.axis, self.begin, self.end,)])
    }

    op_as_typed_op!();
}

impl EvalOp for PulseMask {
    fn is_stateless(&self) -> bool {
        false
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::<PulseMaskOpState>::default()))
    }
}

impl TypedOp for PulseMask {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].clone()))
    }

    as_op!();
}

trivial_op_state_freeeze!(PulseMaskOpState);
