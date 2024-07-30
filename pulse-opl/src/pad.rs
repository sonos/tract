use tract_core::ndarray::*;
use tract_core::ops::array::PadMode;
use tract_nnef::internal::*;
use tract_nnef::ser::tdim;
use tract_nnef::tract_core::ops::OpStateFreeze;

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_pulse_pulse_pad",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Integer.named("axis"),
            TypeName::Integer.named("before"),
            TypeName::Integer.named("after"),
            TypeName::Integer.named("begin_input"),
            TypeName::Integer.named("end_input"),
            TypeName::String.named("border"),
            TypeName::Scalar.named("value"),
            TypeName::Integer.named("overlap"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        deser,
    );
    registry.register_dumper(ser)
}

fn ser(ast: &mut IntoAst, node: &TypedNode, op: &PulsePad) -> TractResult<Option<Arc<RValue>>> {
    let wire = ast.mapping[&node.inputs[0]].clone();
    let dt = ast.model.outlet_fact(node.inputs[0])?.datum_type;
    let (border, value) = tract_nnef::ops::nnef::ser::pad_mode(&op.mode, dt)?;
    let mut params = vec![
        ("axis", numeric(op.axis)),
        ("before", numeric(op.before)),
        ("begin_input", numeric(op.begin_input)),
        ("overlap", numeric(op.overlap)),
        ("after", tdim(&op.after)),
        ("end_input", tdim(&op.end_input)),
    ];
    params.push(("border", string(border)));
    if let Some(value) = value {
        params.push(("value", value));
    }
    Ok(Some(invocation("tract_pulse_pulse_pad", &[wire], &params)))
}

fn deser(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let wire = invocation.named_arg_as(builder, "input")?;
    let axis = invocation.named_arg_as(builder, "axis")?;
    let before = invocation.named_arg_as(builder, "before")?;
    let begin_input = invocation.named_arg_as(builder, "begin_input")?;
    let overlap = invocation.named_arg_as(builder, "overlap")?;
    let border = invocation.named_arg_as::<String>(builder, "border")?;
    let value: Tensor = tensor0(invocation.named_arg_as::<f32>(builder, "value")?);
    let (after, end_input) = builder.allowing_new_symbols(|builder| {
        TractResult::Ok((
            invocation.named_arg_as(builder, "after")?,
            invocation.named_arg_as(builder, "end_input")?,
        ))
    })?;

    let mode = tract_nnef::ops::nnef::deser::pad_mode(&border, value)?;
    let op = PulsePad { axis, before, after, begin_input, end_input, mode, overlap };
    builder.wire(op, &[wire])
}

pub(crate) unsafe fn fill_slice_constant<T: Datum + Copy>(
    data: &mut Tensor,
    constant: &Tensor,
    axis: usize,
    range: std::ops::Range<usize>,
) {
    let c = constant.to_scalar_unchecked::<T>();
    data.to_array_view_mut_unchecked::<T>().slice_axis_mut(Axis(axis), range.into()).fill(*c);
}

unsafe fn fill_slice_with_frame<T: Datum + Copy>(
    data: &mut Tensor,
    axis: usize,
    valid: &Tensor,
    range: std::ops::Range<usize>,
) {
    let mut data = data.to_array_view_mut_unchecked::<T>();
    let valid = valid.to_array_view_unchecked::<T>();
    for i in range {
        data.slice_axis_mut(Axis(axis), (i..i + 1).into()).assign(&valid);
    }
}

#[derive(Debug, Clone, Default, Hash)]
struct PulsePadOpState {
    current_pos: usize,
    last_valid_frame: Option<Tensor>,
}

impl OpState for PulsePadOpState {
    fn eval(
        &mut self,
        session: &mut SessionState,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs).into_tensor();
        let op = op.downcast_ref::<PulsePad>().ok_or_else(|| format_err!("Wrong Op type"))?;
        let tensor = self.pad(session, op, input)?;
        Ok(tvec!(tensor.into_tvalue()))
    }
}

impl PulsePadOpState {
    unsafe fn save_frame<T: Datum + Copy>(&mut self, op: &PulsePad, input: &Tensor, frame: usize) {
        let data = input.to_array_view_unchecked::<T>();
        self.last_valid_frame =
            Some(data.index_axis(Axis(op.axis), frame).to_owned().into_tensor());
    }

    fn pad(
        &mut self,
        session: &SessionState,
        op: &PulsePad,
        mut input: Tensor,
    ) -> TractResult<Tensor> {
        let pulse = input.shape()[op.axis];
        let pulse_begin = self.current_pos;
        let pulse_end = self.current_pos + pulse;
        self.current_pos += pulse - op.overlap;
        let end_input =
            op.end_input.eval(&session.resolved_symbols).to_usize().unwrap_or(usize::MAX);
        let after = op.after.eval(&session.resolved_symbols).to_usize().unwrap_or(usize::MAX);

        if let PadMode::Edge = op.mode {
            if after != 0 && pulse_begin < end_input {
                let latest_valid_frame = (end_input - pulse_begin).min(pulse) - 1;
                unsafe {
                    dispatch_copy_by_size!(Self::save_frame(input.datum_type())(
                        self,
                        op,
                        &input,
                        latest_valid_frame
                    ))
                }
            }
        }

        // pulse is entirely in valid input, just forward
        if pulse_begin >= op.begin_input && pulse_end <= end_input {
            return Ok(input);
        }
        // pulse is entirely before or after output is valid, just forward
        if pulse_end <= op.begin_input - op.before || pulse_begin >= end_input.saturating_add(after)
        {
            return Ok(input);
        }

        if pulse_begin < op.begin_input {
            let fill_up_to = (op.begin_input - pulse_begin).min(pulse);
            match &op.mode {
                PadMode::Constant(c) => unsafe {
                    dispatch_copy_by_size!(fill_slice_constant(input.datum_type())(
                        &mut input,
                        c,
                        op.axis,
                        0..fill_up_to
                    ))
                },
                PadMode::Edge => {
                    let frame = input.slice(op.axis, fill_up_to, fill_up_to + 1)?;
                    unsafe {
                        dispatch_copy_by_size!(fill_slice_with_frame(input.datum_type())(
                            &mut input,
                            op.axis,
                            &frame,
                            0..fill_up_to
                        ))
                    }
                }
                _ => unimplemented!(),
            }
        }
        if pulse_end > end_input && after > 0 {
            let fill_from = pulse - (pulse_end - end_input).min(pulse);
            match &op.mode {
                PadMode::Constant(c) => unsafe {
                    dispatch_copy_by_size!(fill_slice_constant(input.datum_type())(
                        &mut input,
                        c,
                        op.axis,
                        fill_from..pulse
                    ))
                },
                PadMode::Edge => {
                    let last_frame = self.last_valid_frame.as_ref().unwrap();
                    unsafe {
                        dispatch_copy_by_size!(fill_slice_with_frame(input.datum_type())(
                            &mut input,
                            op.axis,
                            last_frame,
                            fill_from..pulse
                        ))
                    }
                }
                _ => unimplemented!(),
            }
        }

        Ok(input)
    }
}

#[derive(Debug, Clone, Default, Hash)]
pub struct PulsePad {
    pub axis: usize,
    pub before: usize,
    pub after: TDim,
    pub begin_input: usize,
    pub end_input: TDim,
    pub mode: PadMode,
    pub overlap: usize,
}

impl Op for PulsePad {
    fn name(&self) -> Cow<str> {
        "PulsePad".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!(
            "Mode: {:?}, axis: {} before: {} after: {}",
            self.mode, self.axis, self.before, self.after,
        )])
    }

    op_as_typed_op!();
}

impl EvalOp for PulsePad {
    fn is_stateless(&self) -> bool {
        false
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::<PulsePadOpState>::default()))
    }
}

impl TypedOp for PulsePad {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].clone()))
    }

    as_op!();
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
struct FrozenPulsePadOpState {
    current_pos: usize,
    last_valid_frame: Option<Arc<Tensor>>,
}

impl OpStateFreeze for PulsePadOpState {
    fn freeze(&self) -> Box<dyn FrozenOpState> {
        Box::new(FrozenPulsePadOpState {
            current_pos: self.current_pos,
            last_valid_frame: self.last_valid_frame.as_ref().map(|t| t.clone().into_arc_tensor()),
        })
    }
}

impl FrozenOpState for FrozenPulsePadOpState {
    fn unfreeze(&self) -> Box<dyn OpState> {
        Box::new(PulsePadOpState {
            current_pos: self.current_pos,
            last_valid_frame: self.last_valid_frame.as_ref().map(|t| t.clone().into_tensor()),
        })
    }
}
