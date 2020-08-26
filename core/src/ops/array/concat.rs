use ndarray::*;

use crate::internal::*;
use crate::pulse::delay::Delay;
use std::ops::Range;

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
tract_linalg::impl_dyn_hash!(TypedConcat);

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
                    inputs[input - 1].shape.dim(self.axis)
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
    not_a_pulsed_op!();
    canonic!();
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
        let dim = inputs.iter().map(|f| f.shape.dim(self.axis)).sum::<TDim>()
            + self
                .slices
                .iter()
                .filter_map(|s| s.as_const())
                .map(|s| s.shape()[self.axis])
                .sum::<usize>();
        fact.shape.set_dim(self.axis, dim)?;
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
                .map(|x| x.to_integer().map(|i| i as usize))
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

    fn pulsify(
        &self,
        source: &TypedModel,
        node: &TypedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
        _pulse: usize,
    ) -> TractResult<TVec<OutletId>> {
        if node.inputs.len() > 1 {
            bail!("Pulsification not implemented for more than one input to Concat")
        }

        let input = mapping[&node.inputs[0]];
        let fact = target.outlet_fact(input)?;

        if fact.axis == self.axis {
            self.pulsify_along_concat_axis(source, node, target, mapping)
        } else {
            bail!("Pulsify for Concat on a separate axis is not implemented (but possible)");
        }
    }
}

impl StatelessOp for TypedConcat {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let refs: TVec<&Tensor> = inputs.iter().map(|i| i.as_ref()).collect();
        let mats = slices(&self.slices, &refs)?;
        let result = Tensor::stack_tensors(self.axis, &mats)?;
        Ok(tvec![result.into_arc_tensor()])
    }
}

impl TypedConcat {
    fn pulsify_along_concat_axis(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        if node.inputs.len() > 1 {
            bail!("Concat can not pulse more than on input on concat axis")
        }
        let mut input = mapping[&node.inputs[0]];
        let fact = target.outlet_fact(input)?.clone();
        assert_eq!(fact.axis, self.axis);
        let var_index = self.slices.iter().position(|s| s.is_var()).unwrap();
        let pre_owned = self.slices[0..var_index]
            .iter()
            .map(|s| s.as_const().unwrap().cast_to_dt(fact.datum_type))
            .collect::<TractResult<TVec<_>>>()?;
        let pre = Tensor::stack_tensors(self.axis, &*pre_owned)?;
        let post_owned = self.slices[var_index + 1..]
            .iter()
            .map(|s| s.as_const().unwrap().cast_to_dt(fact.datum_type))
            .collect::<TractResult<TVec<_>>>()?;
        let post = Tensor::stack_tensors(self.axis, &*post_owned)?;

        let before = pre.shape()[self.axis];
        if fact.delay < before {
            input = target.wire_node(
                format!("{}.Delay", node.name),
                Delay::new(&fact.clone(), before - fact.delay, 0),
                &[input],
            )?[0];
        }
        let main_op = PulsedSameAxisConcat::new(
            self.axis,
            pre,
            post,
            fact.delay.saturating_sub(before),
            fact.dim,
        );
        target.wire_node(&*node.name, main_op, &[input])
    }
}

#[derive(Copy, Clone, Debug)]
enum RangeInRange {
    /// ----[--]<-->HHHH----
    Before(usize),
    /// ----[<----->HHH]HH----
    Begin(usize),
    /// ----[<----->HHHHHHH-]---
    Contain(usize),
    /// ----------<H>[HH]HH----
    Inside(usize),
    /// --------<HHH>[HHHH-]---
    End(usize),
    /// --------HHHHHHH<->[--]---
    After(usize),
}

fn range_in_range(needle: &Range<usize>, haystack: &Range<usize>) -> RangeInRange {
    if needle.end <= haystack.start {
        RangeInRange::Before(haystack.start - needle.end)
    } else if needle.start < haystack.start {
        if needle.end < haystack.end {
            RangeInRange::Begin(haystack.start - needle.start)
        } else {
            RangeInRange::Contain(haystack.start - needle.start)
        }
    } else if needle.start >= haystack.end {
        RangeInRange::After(needle.start - haystack.end)
    } else if needle.end > haystack.end {
        RangeInRange::End(needle.start - haystack.start)
    } else {
        RangeInRange::Inside(needle.start - haystack.start)
    }
}

pub fn overwrite_part_of_pulse<T: Datum>(
    axis: usize,
    pulse_data: &mut Tensor,
    current_pos: usize,
    const_data: &Tensor,
    const_offset: usize,
) -> TractResult<()> {
    let pulse = pulse_data.shape()[axis];
    let const_length = const_data.shape()[axis];
    let const_range = const_offset..const_offset + const_length;
    let pulse_range = current_pos..current_pos + pulse;
    let axis = Axis(axis);
    let mut pulse_data = pulse_data.to_array_view_mut::<T>()?;
    let const_data = const_data.to_array_view::<T>()?;

    match range_in_range(&pulse_range, &const_range) {
        RangeInRange::Before(_) | RangeInRange::After(_) => (),
        RangeInRange::Begin(offset) => {
            // ----[<----->HHH]HH----
            pulse_data
                .slice_axis_mut(axis, (offset..pulse).into())
                .assign(&const_data.slice_axis(axis, (0..pulse - offset).into()));
        }
        RangeInRange::Contain(offset) => {
            // ----[<----->HHHHHHH-]---
            pulse_data
                .slice_axis_mut(axis, (offset..offset + const_length).into())
                .assign(&const_data);
        }
        RangeInRange::Inside(offset) => {
            // ----------<H>[HH]HH----
            pulse_data.assign(&const_data.slice_axis(axis, (offset..offset + pulse).into()));
        }
        RangeInRange::End(offset) => {
            // --------<HHH>[HHHH-]---
            pulse_data
                .slice_axis_mut(axis, (0..const_length - offset).into())
                .assign(&const_data.slice_axis(axis, (offset..const_length).into()));
        }
    }
    Ok(())
}

/// Concat with pulse along concat axis
#[derive(new, Debug, Clone, Hash)]
pub struct PulsedSameAxisConcat {
    axis: usize,
    pre_slice: Tensor,
    post_slice: Tensor,
    input_delay: usize,
    input_len: TDim,
}
tract_linalg::impl_dyn_hash!(PulsedSameAxisConcat);

impl Op for PulsedSameAxisConcat {
    fn name(&self) -> Cow<str> {
        "PulsedSameAxisConcat".into()
    }

    op_core_lir_mir!();
    op_as_typed_op!();
    op_as_pulsed_op!();
}

impl StatefullOp for PulsedSameAxisConcat {
    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        return Ok(Some(Box::new(PulsedSameAxisConcatState::default())));
    }
}

impl TypedOp for PulsedSameAxisConcat {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].clone()))
    }
}

impl PulsedOp for PulsedSameAxisConcat {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        let before = self.pre_slice.shape()[self.axis];
        let after = self.post_slice.shape()[self.axis];
        fact.dim += (before + after).to_dim();
        fact.delay -= before;
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}

#[derive(Clone, Debug, Default)]
pub struct PulsedSameAxisConcatState {
    current_pos: usize,
}

impl OpState for PulsedSameAxisConcatState {
    fn eval(
        &mut self,
        session: &mut SessionState,
        op: &dyn Op,
        mut inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        let op = op.downcast_ref::<PulsedSameAxisConcat>().ok_or("Wrong Op type")?;
        let input = args_1!(inputs);
        let mut data = input.into_tensor();
        let pulse = data.shape()[op.axis];
        let current_pos = self.current_pos;
        self.current_pos += pulse;

        let pre_length = op.pre_slice.shape()[op.axis];
        let pre_offset = op.input_delay - pre_length;
        dispatch_datum!(overwrite_part_of_pulse(data.datum_type())(
            op.axis,
            &mut data,
            current_pos,
            &op.pre_slice,
            pre_offset
        ))?;
        if let Some(l) = session.known_stream_len {
            let input_length = op.input_len.eval(l as _).unwrap() as usize;
            let post_offset = op.input_delay + input_length;
            dispatch_datum!(overwrite_part_of_pulse(data.datum_type())(
                op.axis,
                &mut data,
                current_pos,
                &op.post_slice,
                post_offset
            ))?;
        }

        return Ok(tvec!(data.into_arc_tensor()));
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
