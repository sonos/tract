use crate::ops::prelude::*;
use crate::pulse::delay::Delay;
use ndarray::*;
use num_traits::AsPrimitive;

#[derive(Debug, Clone, PartialEq)]
pub enum PadMode {
    Constant(f32),
    Reflect,
    Edge,
}

impl Default for PadMode {
    fn default() -> PadMode {
        PadMode::Constant(0.0)
    }
}

#[derive(Debug, Clone, new, Default)]
pub struct Pad {
    pads: Vec<(usize, usize)>,
    mode: PadMode,
}

impl Pad {
    fn eval_t<T>(&self, input: SharedTensor) -> TractResult<SharedTensor>
    where
        T: Copy + Datum,
        f32: AsPrimitive<T>,
    {
        let input = input.to_array_view::<T>()?;
        let output_shape: Vec<usize> =
            input.shape().iter().zip(self.pads.iter()).map(|(&d, &(a, b))| d + a + b).collect();
        let element = match self.mode {
            PadMode::Constant(f) => f.as_(),
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
        Ok(output.into())
    }
}

impl Op for Pad {
    fn name(&self) -> Cow<str> {
        "Pad".into()
    }

    fn pulsify(
        &self,
        _source: &NormalizedModel,
        node: &NormalizedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let input = mapping[&node.inputs[0]];
        let input_fact = target.fact(input)?.clone();
        if !self.pads.iter().enumerate().all(|(ax, &(a, b))| ax == input_fact.axis || (a == 0 && b == 0))
        {
            bail!("Pad pulse only implemented for streaming dim");
        }
        if let PadMode::Constant(c) = self.mode {
            let (before, after) = self.pads[input_fact.axis];
            let mut fact = input_fact.clone();
            let mut prec = input;
            if fact.delay < before {
                let buffer_op = Delay::new(fact.clone(), before - fact.delay, 0);
                fact.delay = before;
                let id = target.chain_after(input, format!("{}/Delay", node.name), buffer_op, tvec!(fact.clone()))?;
                prec = OutletId::new(id, 0);
            }
            fact.dim += (before + after).to_dim();
            fact.delay -= before;
            let op = PulsePad::<f32>::new(
                input_fact.axis,
                input_fact.pulse(),
                input_fact.delay + before,
                (input_fact.delay + before).to_dim() + input_fact.dim,
                c,
            );
            let id = target.chain_after(prec, &*node.name, op, tvec!(fact))?;

            Ok(tvec!(OutletId::new(id,0)))
        } else {
            bail!("Pad pulse only implemented for constant");
        }
    }
}

impl StatelessOp for Pad {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
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
}

#[derive(Debug, Clone, Default, new)]
struct PulsePadOpState<T: Datum + Copy> {
    current_pos: usize,
    _slimer: PhantomData<T>,
}

impl<T: Datum + Copy> OpState for PulsePadOpState<T> {
    fn eval(
        &mut self,
        session: &mut SessionState,
        op: &Op,
        mut inputs: TVec<SharedTensor>,
    ) -> TractResult<TVec<SharedTensor>> {
        let input = args_1!(inputs);
        let op = op.downcast_ref::<PulsePad<T>>().ok_or("Wrong Op type")?;
        let current_pos = self.current_pos;
        self.current_pos += op.pulse;
        // pulse is entirely before or after input, emit padding constant
        // (if the session has not seen the end of input stream, then
        // we can't be processing it yet)
        if current_pos + op.pulse <= op.begin_input
            || session
                .known_stream_len
                .map(|s| current_pos >= op.end_input.eval(s as i32).unwrap() as usize)
                .unwrap_or(false)
        {
            return Ok(tvec!(ArrayD::from_elem(input.shape(), op.constant).into()));
        }
        let mut data = input.to_tensor().into_array::<T>()?;
        if current_pos < op.begin_input {
            data.slice_axis_mut(Axis(op.axis), (0..op.begin_input - current_pos).into()).fill(op.constant);
        }
        if let Some(s) = session.known_stream_len {
            let end_input = op.end_input.eval(s as i32).unwrap() as usize;
            if current_pos + op.pulse > end_input {
                data.slice_axis_mut(Axis(op.axis), (end_input - current_pos..op.pulse).into()).fill(op.constant);
            }
        }
        Ok(tvec!(data.into()))
    }
}

#[derive(Debug, Clone, Default, new)]
struct PulsePad<T: Datum + Copy> {
    axis: usize,
    pulse: usize,
    begin_input: usize,
    end_input: TDim,
    constant: T,
}

impl<T: Datum + Copy> Op for PulsePad<T> {
    fn name(&self) -> Cow<str> {
        "Pad".into()
    }
}

impl<T: Datum + Copy> StatefullOp for PulsePad<T> {
    fn state(&self) -> TractResult<Option<Box<OpState>>> {
        Ok(Some(Box::new(PulsePadOpState::<T>::default())))
    }
}

impl<T: Datum + Copy> InferenceRulesOp for PulsePad<T> {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        _s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        Ok(())
    }
}
