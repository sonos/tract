use tract_nnef::internal::*;
use tract_core::ndarray::*;
use tract_core::ops::array::PadMode;

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
        mut inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs).into_tensor();
        let op = op.downcast_ref::<PulsePad>().ok_or_else(|| format_err!("Wrong Op type"))?;
        let tensor = self.pad(session, op, input)?;
        Ok(tvec!(tensor.into_arc_tensor()))
    }
}

impl PulsePadOpState {
    unsafe fn save_frame<T: Datum + Copy>(&mut self, op: &PulsePad, input: &Tensor, frame: usize) {
        let data = input.to_array_view_unchecked::<T>();
        self.last_valid_frame =
            Some(data.index_axis(Axis(op.axis), frame).to_owned().into_tensor());
    }

    unsafe fn fill_slice_constant<T: Datum + Copy>(
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

    fn pad(
        &mut self,
        session: &mut SessionState,
        op: &PulsePad,
        mut input: Tensor,
    ) -> TractResult<Tensor> {
        let pulse_begin = self.current_pos;
        let pulse_end = self.current_pos + op.pulse;
        self.current_pos += op.pulse;
        let end_input =
            op.end_input.eval(&session.resolved_symbols).to_usize().unwrap_or(std::usize::MAX);
        let after = op.after.eval(&session.resolved_symbols).to_usize().unwrap_or(std::usize::MAX);

        if let PadMode::Edge = op.mode {
            if after != 0 && pulse_begin < end_input {
                let latest_valid_frame = (end_input - pulse_begin).min(op.pulse) - 1;
                unsafe {
                    tract_core::dispatch_copy_by_size!(Self::save_frame(input.datum_type())(
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
            let fill_up_to = (op.begin_input - pulse_begin).min(op.pulse);
            match &op.mode {
                PadMode::Constant(c) => unsafe {
                    tract_core::dispatch_copy_by_size!(Self::fill_slice_constant(
                        input.datum_type()
                    )(
                        &mut input, c, op.axis, 0..fill_up_to
                    ))
                },
                PadMode::Edge => {
                    let frame = input.slice(op.axis, fill_up_to, fill_up_to + 1)?;
                    unsafe {
                        tract_core::dispatch_copy_by_size!(Self::fill_slice_with_frame(
                            input.datum_type()
                        )(
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
            let fill_from = op.pulse - (pulse_end - end_input).min(op.pulse);
            match &op.mode {
                PadMode::Constant(c) => unsafe {
                    tract_core::dispatch_copy_by_size!(Self::fill_slice_constant(
                        input.datum_type()
                    )(
                        &mut input,
                        c,
                        op.axis,
                        fill_from..op.pulse
                    ))
                },
                PadMode::Edge => {
                    let last_frame = self.last_valid_frame.as_ref().unwrap();
                    unsafe {
                        tract_core::dispatch_copy_by_size!(Self::fill_slice_with_frame(
                            input.datum_type()
                        )(
                            &mut input,
                            op.axis,
                            last_frame,
                            fill_from..op.pulse
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
    pub pulse: usize,
    pub before: usize,
    pub after: TDim,
    pub begin_input: usize,
    pub end_input: TDim,
    pub mode: PadMode,
}

tract_data::impl_dyn_hash!(PulsePad);

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

    op_pulse!();
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
        Ok(Some(Box::new(PulsePadOpState::default())))
    }
}

impl TypedOp for PulsePad {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].clone()))
    }

    as_op!();
}
