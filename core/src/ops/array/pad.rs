use crate::internal::*;

/// How a padded region is filled. A `Constant` value carries a single element
/// in the datum type the model builder gave it, f32 for a float graph, and no
/// pass retypes it: a consumer casts it to the datum type of the tensor it pads.
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

#[derive(Debug, Clone, new, Default, Hash, PartialEq, Eq)]
pub struct Pad {
    pub pads: Vec<(usize, usize)>,
    pub mode: PadMode,
}

impl Pad {
    fn eval_t<T>(&self, input_tensor: TValue) -> TractResult<TValue>
    where
        T: Copy + Datum + num_traits::Zero + num_traits::ToPrimitive,
    {
        use tract_ndarray::*;
        let input = input_tensor.to_plain_array_view::<T>()?;
        let output_shape: Vec<usize> =
            input.shape().iter().zip(self.pads.iter()).map(|(&d, &(a, b))| d + a + b).collect();
        let element = match &self.mode {
            PadMode::Constant(f) => f.cast_to_scalar::<T>()?,
            _ => T::default(),
        };
        let mut output = if matches!(self.mode, PadMode::Constant(_))
            && let Some(source) = input.as_slice()
        {
            let mut output = Tensor::zero::<T>(&output_shape)?;
            if element != T::zero() || element.to_f64().is_some_and(f64::is_sign_negative) {
                output.fill_t(element)?;
            }
            self.copy_constant_rows(
                input.shape(),
                source,
                &output_shape,
                output.try_as_plain_ram_mut()?.as_slice_mut::<T>()?,
            )?;
            output
        } else {
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
            output.into_tensor()
        };
        unsafe { output.set_datum_type(input_tensor.datum_type()) }
        Ok(output.into_tvalue())
    }

    /// Copies contiguous input runs into constant-filled rows, splitting only at row boundaries.
    fn copy_constant_rows<T: Datum + Copy>(
        &self,
        input_shape: &[usize],
        source: &[T],
        output_shape: &[usize],
        output: &mut [T],
    ) -> TractResult<()> {
        if source.is_empty() || output.is_empty() {
            return Ok(());
        }
        let axis = self.pads.iter().rposition(|p| *p != (0, 0)).unwrap_or(0);
        let row_len: usize = output_shape[axis..].iter().product();
        let copy_len: usize = input_shape[axis..].iter().product();
        let inner: usize = input_shape.get(axis + 1..).unwrap_or(&[]).iter().product();
        let start = self.pads.get(axis).map_or(0, |p| p.0) * inner;
        tract_linalg::multithread::par_chunks_mut(output, row_len, source.len(), |first, chunk| {
            for (ix, row) in chunk.chunks_mut(row_len).enumerate() {
                let mut coord = first + ix;
                let mut source_row = 0;
                let mut stride = 1;
                let mut inside = true;
                for ax in (0..axis).rev() {
                    let output_coord = coord % output_shape[ax];
                    coord /= output_shape[ax];
                    let Some(input_coord) =
                        output_coord.checked_sub(self.pads[ax].0).filter(|c| *c < input_shape[ax])
                    else {
                        inside = false;
                        break;
                    };
                    source_row += input_coord * stride;
                    stride *= input_shape[ax];
                }
                if inside {
                    let from = source_row * copy_len;
                    row[start..][..copy_len].copy_from_slice(&source[from..][..copy_len]);
                }
            }
            Ok(())
        })
    }
}

impl Op for Pad {
    fn name(&self) -> StaticName {
        "Pad".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("Mode: {:?}, pads: {:?})", self.mode, self.pads,)])
    }

    op_as_typed_op!();
}

impl EvalOp for Pad {
    op_out_of_plan!();

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
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

    fn input_roi(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TVec<Option<TDim>>>> {
        let output_fact = model.outlet_fact(OutletId::new(node.id, 0))?;
        rule_if_some!(roi = &output_fact.region_of_interest);
        // For each padded axis, substitute 🎯axis → 🎯axis - before
        let mut input_roi = roi.clone();
        for (axis, &(before, _)) in self.pads.iter().enumerate() {
            if before == 0 {
                continue;
            }
            if let Some(sym) = input_roi
                .symbols()
                .into_iter()
                .find(|s| crate::ops::logic::sym_to_coord_axis(s) == Some(axis))
            {
                let shifted = TDim::Sym(sym.clone()) - TDim::Val(before as i64);
                input_roi = input_roi.substitute(&sym, &shifted).unwrap_or(input_roi);
            }
        }
        Ok(Some(tvec![Some(input_roi)]))
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
        if let (InOut::In(0), AxisOp::Rm(ix)) = (io, change)
            && new_op.pads.remove(*ix) == (0, 0)
        {
            return Ok(Some(AxisChangeConsequence::new(
                model,
                node,
                Some(Box::new(new_op)),
                change,
            )));
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
