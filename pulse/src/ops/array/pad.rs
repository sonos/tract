use crate::internal::*;
use tract_core::ndarray::*;
use tract_core::ops::array::{Pad, PadMode};
use tract_pulse_opl::ops::{Delay, PulsePad};

submit_op_pulsifier!(Pad, pulsify);

fn pulsify(
    op: &Pad,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _pulse: usize,
) -> TractResult<TVec<OutletId>> {
    let mut input = mapping[&node.inputs[0]];
    let fact = target.outlet_fact(input)?.clone();
    if !op.pads.iter().enumerate().all(|(ax, &(a, b))| ax == fact.axis || (a == 0 && b == 0)) {
        bail!("Pad pulse only implemented for streaming dim");
    }
    let (before, after) = op.pads[fact.axis];
    let pulse = fact.pulse();
    let mut extra_delay = before.saturating_sub(fact.delay);
    match op.mode {
        PadMode::Constant(_) => (),
        PadMode::Edge if before < pulse => {
            let start_offset = (fact.delay + extra_delay) % pulse;
            if before > start_offset {
                extra_delay += before - start_offset;
            }
        }
        PadMode::Edge => bail!(
            "Edge padding mode needs pulse strictly bigger than left padding (pulse={} padding={})",
            pulse,
            before
        ),
        PadMode::Reflect => bail!("Reflect padding mode pulsing is not supported"),
    };
    if extra_delay > 0 {
        input = target.wire_node(
            format!("{}.Delay", node.name),
            Delay::new(fact.axis, &(&fact).into(), extra_delay, 0),
            &[input],
        )?[0];
    }
    let op = PulsePad {
        axis: fact.axis,
        pulse,
        before,
        after: after.into(),
        begin_input: fact.delay + extra_delay,
        end_input: fact.delay.to_dim() + extra_delay + fact.dim,
        mode: op.mode.clone(),
    };
    target.wire_node(&*node.name, op, &[input])
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
            let fill_up_to = (op.begin_input - pulse_begin).min(op.pulse);
            match &op.mode {
                PadMode::Constant(c) => unsafe {
                    dispatch_copy_by_size!(Self::fill_slice_constant(input.datum_type())(
                        &mut input,
                        &c,
                        op.axis,
                        0..fill_up_to
                    ))
                },
                PadMode::Edge => {
                    let frame = input.slice(op.axis, fill_up_to, fill_up_to + 1)?;
                    unsafe {
                        dispatch_copy_by_size!(Self::fill_slice_with_frame(input.datum_type())(
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
                    dispatch_copy_by_size!(Self::fill_slice_constant(input.datum_type())(
                        &mut input,
                        c,
                        op.axis,
                        fill_from..op.pulse
                    ))
                },
                PadMode::Edge => {
                    let last_frame = self.last_valid_frame.as_ref().unwrap();
                    unsafe {
                        dispatch_copy_by_size!(Self::fill_slice_with_frame(input.datum_type())(
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

impl PulsedOp for PulsePad {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        fact.dim += self.before.to_dim() + &self.after;
        fact.delay -= self.before;
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
