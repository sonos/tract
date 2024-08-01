use crate::internal::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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

impl Pad {
    fn eval_t<T>(&self, input_tensor: TValue) -> TractResult<TValue>
    where
        T: Copy + Datum,
    {
        use tract_ndarray::*;
        let input = input_tensor.to_array_view::<T>()?;
        let output_shape: Vec<usize> =
            input.shape().iter().zip(self.pads.iter()).map(|(&d, &(a, b))| d + a + b).collect();
        let element = match &self.mode {
            PadMode::Constant(f) => f.cast_to_scalar::<T>()?,
            _ => T::default(),
        };
        let mut output = ArrayD::<T>::from_elem(output_shape, element);
        let slice_spec: Vec<SliceInfoElem> = self
            .pads
            .iter()
            .map(|&(a, b)| SliceInfoElem::Slice {
                start: a as isize,
                end: if b != 0 { Some(-(b as isize)) } else { None },
                step: 1,
            })
            .collect();
        let slice_info = SliceInfo::<_, IxDyn, IxDyn>::try_from(slice_spec).unwrap();
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
        let mut output = output.into_tensor();
        unsafe { output.set_datum_type(input_tensor.datum_type()) }
        Ok(output.into_tvalue())
    }
}

impl Op for Pad {
    fn name(&self) -> Cow<str> {
        "Pad".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("Mode: {:?}, pads: {:?})", self.mode, self.pads,)])
    }

    op_as_typed_op!();
}

impl EvalOp for Pad {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        Ok(tvec!(dispatch_numbers!(Self::eval_t(input.datum_type())(self, input))?))
    }
}

impl TypedOp for Pad {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = inputs[0].without_value();
        if self.pads.len() != fact.rank() {
            bail!("Inconsistent pad: input of rank {}, pads are: {:?}", fact.rank(), self.pads);
        }
        for (ix, (b, e)) in self.pads.iter().enumerate() {
            fact.shape.set(ix, fact.shape[ix].clone() + *b + *e);
        }
        Ok(tvec!(fact))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        let mut result = AxesMapping::disconnected(inputs, outputs)?;
        for (ix, pads) in self.pads.iter().enumerate() {
            if pads == &(0, 0) {
                result = result.linking((InOut::In(0), ix), (InOut::Out(0), ix))?;
            }
        }
        Ok(result)
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        let mut new_op = self.clone();
        if let (InOut::In(0), AxisOp::Rm(ix)) = (io, change) {
            if new_op.pads.remove(*ix) == (0, 0) {
                return Ok(Some(AxisChangeConsequence::new(
                    model,
                    node,
                    Some(Box::new(new_op)),
                    change,
                )));
            }
        }
        if let (InOut::In(0), AxisOp::Add(ix)) = (io, change) {
            new_op.pads.insert(*ix, (0, 0));
            return Ok(Some(AxisChangeConsequence::new(
                model,
                node,
                Some(Box::new(new_op)),
                change,
            )));
        }
        Ok(None)
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if self.pads.iter().all(|p| p.0 == 0 && p.1 == 0) {
            TypedModelPatch::shunt_one_op(model, node)
        } else {
            Ok(None)
        }
    }
}
