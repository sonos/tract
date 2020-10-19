use crate::internal::*;
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
tract_data::impl_dyn_hash!(Pad);

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

    op_core_lir_mir!();
    op_as_typed_op!();
}

impl EvalOp for Pad {
    fn is_stateless(&self) -> bool {
        true
    }

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
            fact.shape[ix] = fact.shape[ix].clone() + *b + *e;
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
}
