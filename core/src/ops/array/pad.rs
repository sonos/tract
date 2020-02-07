use crate::internal::*;
use crate::infer::*;
use crate::pulse::delay::Delay;
use ndarray::*;

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, new, Default)]
pub struct Pad {
    pads: Vec<(usize, usize)>,
    mode: PadMode,
}

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
        Ok(vec!(format!(
            "Mode: {:?}, pads: {:?})",
            self.mode,
            self.pads,
        )))
    }

    canonic!();
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

impl InferenceRulesOp for Pad {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        for (ix, &(a, b)) in self.pads.iter().enumerate() {
            s.equals(&inputs[0].shape[ix], outputs[0].shape[ix].bex() - a.to_dim() - b.to_dim())?;
        }
        Ok(())
    }

    inference_op_as_op!();
    to_typed!();
}

impl TypedOp for Pad {
    typed_op_as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = inputs[0].clone();
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
        if self.pads.iter().all(|p| p.0 == 0 && p.1 == 0 ) {
            Ok(Some(TypedModelPatch::shunt_one_op(model, node)?))
        } else {
            Ok(None)
        }
    }

    fn pulsify(
        &self,
        _source: &NormalizedModel,
        node: &NormalizedNode,
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
                format!("{}/Delay", node.name),
                Delay::new(&fact.clone(), extra_delay, 0),
                &[input],
            )?[0];
        }
        let op = PulsePad::<f32>::new(
            fact.axis,
            pulse,
            before,
            after,
            fact.delay + extra_delay,
            fact.delay.to_dim() + extra_delay + fact.dim,
            self.mode.clone(),
        );
        target.wire_node(&*node.name, op, &[input])
    }
}

#[derive(Debug, Clone, Default, new)]
struct PulsePadOpState<T: Datum + Copy> {
    current_pos: usize,
    last_valid_frame: Option<Tensor>,
    _slimer: PhantomData<T>,
}

impl<T: Datum + Copy> OpState for PulsePadOpState<T> {
    fn eval(
        &mut self,
        session: &mut SessionState,
        op: &dyn Op,
        mut inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        let op = op.downcast_ref::<PulsePad<T>>().ok_or("Wrong Op type")?;
        let pulse_begin = self.current_pos;
        let pulse_end = self.current_pos + op.pulse;
        self.current_pos += op.pulse;
        let end_input = session
            .known_stream_len
            .map(|s| op.end_input.eval(s as i32).unwrap() as usize)
            .unwrap_or(std::usize::MAX);

        if let PadMode::Edge = op.mode {
            if op.after > 0 && pulse_begin < end_input {
                let latest_valid_frame = (end_input - pulse_begin).min(op.pulse) - 1;
                let data = inputs[0].to_array_view::<T>()?;
                self.last_valid_frame = Some(
                    data.index_axis(Axis(op.axis), latest_valid_frame).to_owned().into_tensor(),
                );
            }
        }

        // pulse is entirely in valid input, just forward
        if pulse_begin >= op.begin_input && pulse_end <= end_input {
            return Ok(inputs);
        }
        // pulse is entirely before or after output is valid, just forward
        if pulse_end <= op.begin_input - op.before
            || pulse_begin >= end_input.saturating_add(op.after)
        {
            return Ok(inputs);
        }
        let input = args_1!(inputs);
        let mut data = input.into_tensor().into_array::<T>()?;

        if pulse_begin < op.begin_input {
            let fill_up_to = (op.begin_input - pulse_begin).min(op.pulse);
            match &op.mode {
                PadMode::Constant(c) => {
                    let c = c.to_scalar::<T>()?;
                    data.slice_axis_mut(Axis(op.axis), (0..fill_up_to).into()).fill(*c);
                }
                PadMode::Edge => {
                    let data = data.view_mut();
                    let (mut padding, valid) = data.split_at(Axis(op.axis), fill_up_to);
                    let first_frame = valid.index_axis_move(Axis(op.axis), 0);
                    for i in 0..fill_up_to {
                        padding.index_axis_mut(Axis(op.axis), i).assign(&first_frame);
                    }
                }
                _ => unimplemented!(),
            }
        }
        if pulse_end > end_input && op.after > 0 {
            let fill_from = op.pulse - (pulse_end - end_input).min(op.pulse);
            match &op.mode {
                PadMode::Constant(c) => {
                    let c = c.to_scalar::<T>()?;
                    data.slice_axis_mut(Axis(op.axis), (fill_from..op.pulse).into()).fill(*c);
                }
                PadMode::Edge => {
                    let mut data = data.view_mut();
                    let last_frame =
                        self.last_valid_frame.as_ref().unwrap().to_array_view::<T>()?;
                    for i in fill_from..op.pulse {
                        data.index_axis_mut(Axis(op.axis), i).assign(&last_frame);
                    }
                }
                _ => unimplemented!(),
            }
        }

        Ok(tvec!(data.into_arc_tensor()))
    }
}

#[derive(Debug, Clone, Default, new)]
struct PulsePad<T: Datum + Copy> {
    axis: usize,
    pulse: usize,
    before: usize,
    after: usize,
    begin_input: usize,
    end_input: TDim,
    mode: PadMode,
    _slimer: PhantomData<T>,
}

impl<T: Datum + Copy> Op for PulsePad<T> {
    fn name(&self) -> Cow<str> {
        "PulsePad".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec!(format!(
            "Mode: {:?}, axis: {} before: {} after: {}",
            self.mode,
            self.axis,
            self.before,
            self.after,
        )))
    }

    canonic!();
    op_as_typed_op!();
    op_as_pulsed_op!();
}

impl<T: Datum + Copy> StatefullOp for PulsePad<T> {
    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(PulsePadOpState::<T>::default())))
    }
}

impl<T: Datum + Copy> TypedOp for PulsePad<T> {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].clone()))
    }

    typed_op_as_op!();
}

impl<T: Datum + Copy> PulsedOp for PulsePad<T> {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        fact.dim += (self.before + self.after).to_dim();
        fact.delay -= self.before;
        Ok(tvec!(fact))
    }

    pulsed_op_as_op!();
    pulsed_op_to_typed_op!();
}
