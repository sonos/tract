use crate::internal::*;
use crate::pulse::delay::Delay;
use ndarray::*;

#[derive(Debug, Clone, PartialEq, Hash)]
pub enum PadMode {
    Constant(Arc<Tensor>),
    Reflect,
    Edge,
}

impl Default for PadMode {
    fn default() -> PadMode {
        PadMode::Constant(Arc::new(0.0f32.into()))
    }
}

#[derive(Debug, Clone, new, Default, Hash)]
pub struct Pad {
    pub pads: Vec<(usize, usize)>,
    pub mode: PadMode,
}
tract_linalg::impl_dyn_hash!(Pad);

impl Pad {
    fn eval_t<T>(&self, input: Arc<Tensor>) -> TractResult<Arc<Tensor>>
    where
        T: Copy + Datum,
    {
        let input = input.to_array_view::<T>()?;
        let output_shape: Vec<usize> =
            input.shape().iter().zip(self.pads.iter()).map(|(&d, &(a, b))| d + a + b).collect();
        let element = match &self.mode {
            PadMode::Constant(f) => f.to_scalar::<T>()?.clone(),
            _ => T::default(),
        };
        let mut output = ArrayD::<T>::from_elem(output_shape, element);
        let slice_spec: Vec<SliceOrIndex> = self
            .pads
            .iter()
            .map(|&(a, b)| SliceOrIndex::Slice {
                start: a as isize,
                end: if b != 0 { Some(-(b as isize)) } else { None },
                step: 1,
            })
            .collect();
        let slice_info = SliceInfo::<_, IxDyn>::new(slice_spec).unwrap();
        output.slice_mut(slice_info.as_ref()).assign(&input);
        if self.mode == PadMode::Reflect || self.mode == PadMode::Edge {
            for (ax, &(bef, aft)) in self.pads.iter().enumerate() {
                let axis = Axis(ax);
                let dim = output.shape()[ax];
                {
                    let (mut pad, data) = output.view_mut().split_at(axis, bef);
                    for i in 0..bef {
                        let mut target = pad.slice_axis_mut(axis, Slice::from(i..i + 1));
                        let source_slice = match self.mode {
                            PadMode::Edge => 0,
                            PadMode::Reflect => bef - i,
                            _ => panic!(),
                        };
                        let source =
                            data.slice_axis(axis, Slice::from(source_slice..source_slice + 1));
                        target.assign(&source);
                    }
                }
                {
                    let (data, mut pad) = output.view_mut().split_at(axis, dim - aft);
                    for i in 0..aft {
                        let mut target = pad.slice_axis_mut(axis, Slice::from(i..i + 1));
                        let source_slice = match self.mode {
                            PadMode::Edge => dim - aft - 1,
                            PadMode::Reflect => dim - aft - 2 - i,
                            _ => panic!(),
                        };
                        let source =
                            data.slice_axis(axis, Slice::from(source_slice..source_slice + 1));
                        target.assign(&source);
                    }
                }
            }
        }
        Ok(output.into_arc_tensor())
    }
}

impl Op for Pad {
    fn name(&self) -> Cow<str> {
        "Pad".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("Mode: {:?}, pads: {:?})", self.mode, self.pads,)])
    }

    canonic!();
    op_core_lir_mir!();
    op_as_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for Pad {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        Ok(tvec!(dispatch_numbers!(Self::eval_t(input.datum_type())(self, input))?))
    }
}

impl TypedOp for Pad {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = inputs[0].clone();
        if self.pads.len() != fact.rank() {
            bail!("Inconsistent pad: input of rank {}, pads are: {:?}", fact.rank(), self.pads);
        }
        for (ix, (b, e)) in self.pads.iter().enumerate() {
            fact.shape.set_dim(ix, fact.shape.dim(ix).clone() + *b + *e)?
        }
        Ok(tvec!(fact))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if self.pads.iter().all(|p| p.0 == 0 && p.1 == 0) {
            Ok(Some(TypedModelPatch::shunt_one_op(model, node)?))
        } else {
            Ok(None)
        }
    }

    fn pulsify(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
        _pulse: usize,
    ) -> TractResult<TVec<OutletId>> {
        let mut input = mapping[&node.inputs[0]];
        let fact = target.outlet_fact(input)?.clone();
        if !self.pads.iter().enumerate().all(|(ax, &(a, b))| ax == fact.axis || (a == 0 && b == 0))
        {
            bail!("Pad pulse only implemented for streaming dim");
        }
        let (before, after) = self.pads[fact.axis];
        let pulse = fact.pulse();
        let mut extra_delay = before.saturating_sub(fact.delay);
        match self.mode {
            PadMode::Constant(_) => (),
            PadMode::Edge if before < pulse => {
                let start_offset = (fact.delay + extra_delay) % pulse;
                if before > start_offset {
                    extra_delay += before - start_offset;
                }
            },
            PadMode::Edge => bail!("Edge padding mode needs pulse strictly bigger than left padding (pulse={} padding={})", pulse, before),
            PadMode::Reflect => bail!("Reflect padding mode pulsing is not supported")
        };
        if extra_delay > 0 {
            input = target.wire_node(
                format!("{}.Delay", node.name),
                Delay::new(&fact.clone(), extra_delay, 0),
                &[input],
            )?[0];
        }
        let op = PulsePad::new(
            fact.axis,
            pulse,
            before,
            after.into(),
            fact.delay + extra_delay,
            fact.delay.to_dim() + extra_delay + fact.dim,
            self.mode.clone(),
        );
        target.wire_node(&*node.name, op, &[input])
    }
}

#[derive(Debug, Clone, Default, new, Hash)]
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
        let op = op.downcast_ref::<PulsePad>().ok_or("Wrong Op type")?;
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
        let end_input = session
            .known_stream_len
            .map(|s| op.end_input.eval(s as _).unwrap() as usize)
            .unwrap_or(std::usize::MAX);
        let after = session
            .known_stream_len
            .map(|s| op.after.eval(s as _).unwrap() as usize)
            .unwrap_or(std::usize::MAX);

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
                        c,
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

#[derive(Debug, Clone, Default, new, Hash)]
pub struct PulsePad {
    pub axis: usize,
    pub pulse: usize,
    pub before: usize,
    pub after: TDim,
    pub begin_input: usize,
    pub end_input: TDim,
    pub mode: PadMode,
}

tract_linalg::impl_dyn_hash!(PulsePad);

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

    canonic!();
    op_core_lir_mir!();
    op_as_typed_op!();
    op_as_pulsed_op!();
}

impl StatefullOp for PulsePad {
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
