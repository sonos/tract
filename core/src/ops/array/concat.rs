use crate::internal::*;

/// ConcatSlice: fully decluttered Concat equivalent
#[derive(Debug, Clone, Hash)]
pub enum ConcatSlice {
    Const(Arc<Tensor>),
    Var,
}

impl ConcatSlice {
    pub fn as_const(&self) -> Option<&Tensor> {
        match self {
            ConcatSlice::Const(c) => Some(&c),
            ConcatSlice::Var => None,
        }
    }

    pub fn is_var(&self) -> bool {
        match self {
            ConcatSlice::Const(_) => false,
            ConcatSlice::Var => true,
        }
    }
}

#[derive(new, Debug, Clone, Hash)]
pub struct TypedConcat {
    pub axis: usize,
    pub slices: TVec<ConcatSlice>,
}
tract_data::impl_dyn_hash!(TypedConcat);

impl TypedConcat {
    pub fn concat_vars(axis: usize, n: usize) -> TypedConcat {
        TypedConcat { axis, slices: std::iter::repeat(ConcatSlice::Var).take(n).collect() }
    }

    pub fn offsets(&self, inputs: &[&TypedFact]) -> TractResult<Vec<TDim>> {
        let mut offsets = vec![0.to_dim()];
        let mut input = 0;
        for slice in &self.slices {
            let len = match slice {
                ConcatSlice::Const(t) => t.shape()[self.axis].to_dim(),
                ConcatSlice::Var => {
                    input += 1;
                    inputs[input - 1].shape[self.axis].clone()
                }
            };
            let offset = len + offsets.last().unwrap();
            offsets.push(offset)
        }
        Ok(offsets)
    }
}

impl Op for TypedConcat {
    fn name(&self) -> Cow<str> {
        "Concat".into()
    }

    op_core_lir_mir!();
    op_as_typed_op!();
}

impl TypedOp for TypedConcat {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = inputs
            .get(0)
            .cloned()
            .cloned()
            .or_else(|| {
                if let ConcatSlice::Const(t) = &self.slices[0] {
                    Some(TypedFact::dt_shape(t.datum_type(), t.shape()).unwrap())
                } else {
                    None
                }
            })
            .unwrap();
        for input in inputs {
            if input.rank() != fact.rank()
                || input
                    .shape
                    .iter()
                    .zip(fact.shape.iter())
                    .enumerate()
                    .filter(|(ax, _)| *ax != self.axis)
                    .any(|(_, (i, f))| i != f)
            {
                bail!("Inconsistent concat {:?} inputs: {:?}", self, inputs);
            }
        }
        let dim = inputs.iter().map(|f| &f.shape[self.axis]).sum::<TDim>()
            + self
                .slices
                .iter()
                .filter_map(|s| s.as_const())
                .map(|s| s.shape()[self.axis])
                .sum::<usize>();
        fact.shape[self.axis] = dim;
        Ok(tvec!(fact))
    }

    fn invariants(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Invariants> {
        if self.slices.iter().any(|s| s.as_const().is_some()) {
            Ok(Invariants::none())
        } else {
            let rank = model.outlet_fact(node.inputs[0])?.shape.rank();
            (0..rank)
                .filter(|&ax| ax != self.axis)
                .map(|axis| AxisInfo::for_node(model, node, axis))
                .collect()
        }
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        let axis =
            if let Some(axis) = change.transform_axis(self.axis) { axis } else { return Ok(None) };
        let op = TypedConcat {
            axis,
            slices: self
                .slices
                .iter()
                .map(|s| match s {
                    ConcatSlice::Var => Ok(ConcatSlice::Var),
                    ConcatSlice::Const(c) => {
                        let mut c = c.clone().into_tensor();
                        change.change_tensor(&mut c)?;
                        Ok(ConcatSlice::Const(c.into_arc_tensor()))
                    }
                })
                .collect::<TractResult<_>>()?,
        };
        Ok(Some(AxisChangeConsequence::new(model, node, Some(Box::new(op)), change)))
    }

    fn slice_output(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        patch: &mut TypedModelPatch,
        _output_slot: usize,
        axis: usize,
        start: usize,
        end: usize,
    ) -> TractResult<Option<OutletId>> {
        let inputs = model.node_input_facts(node.id)?;
        if self.axis == axis {
            let mut input = 0;
            let offsets = self
                .offsets(&inputs)?
                .iter()
                .map(|x| x.to_usize())
                .collect::<TractResult<Vec<usize>>>()?;
            for (ix, slice) in self.slices.iter().enumerate() {
                if start >= offsets[ix] && end <= offsets[ix + 1] {
                    match slice {
                        ConcatSlice::Const(t) => {
                            return Ok(Some(patch.add_const(
                                format!("{}-const", node.name),
                                t.slice(axis, start - offsets[ix], end - offsets[ix])?,
                            )?))
                        }
                        ConcatSlice::Var => {
                            let prec = model.node(node.inputs[input].node);
                            return prec.op().as_typed().unwrap().slice_output(
                                model,
                                &prec,
                                patch,
                                node.inputs[input].slot,
                                axis,
                                start - offsets[ix],
                                end - offsets[ix],
                            );
                        }
                    };
                }
                input += slice.is_var() as usize;
            }
        }
        Ok(None)
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        for (ix, outlet) in node.inputs.iter().enumerate() {
            if let Some(konst) = model.outlet_fact(*outlet)?.konst.as_ref() {
                let slice_position =
                    self.slices.iter().enumerate().filter(|(_, s)| s.is_var()).nth(ix).unwrap().0;
                let op = TypedConcat {
                    axis: self.axis,
                    slices: self
                        .slices
                        .iter()
                        .enumerate()
                        .map(|(ix, slice)| {
                            if slice_position == ix {
                                ConcatSlice::Const(konst.clone())
                            } else {
                                slice.clone()
                            }
                        })
                        .collect(),
                };
                let mut inputs = node.inputs.to_vec();
                inputs.remove(ix);
                return Ok(Some(TypedModelPatch::replace_single_op(model, node, &*inputs, op)?))
            }
        }
        Ok(None)
    }
}

impl EvalOp for TypedConcat {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let refs: TVec<&Tensor> = inputs.iter().map(|i| i.as_ref()).collect();
        let mats = slices(&self.slices, &refs)?;
        let result = Tensor::stack_tensors(self.axis, &mats)?;
        Ok(tvec![result.into_arc_tensor()])
    }
}

////////////////////////////////////////////////

fn slices<'a, 'i: 'a, 'o: 'a>(
    slices: &'o [ConcatSlice],
    inputs: &'i [&'i Tensor],
) -> TractResult<TVec<&'a Tensor>> {
    let mut mats: TVec<&'a Tensor> = tvec![];
    let mut input_idx = 0;
    for slice in slices {
        match slice {
            ConcatSlice::Const(c) => mats.push(c),
            ConcatSlice::Var => {
                let inp_view = inputs[input_idx];
                mats.push(inp_view);
                input_idx += 1
            }
        }
    }
    Ok(mats)
}
