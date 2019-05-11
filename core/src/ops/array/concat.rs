use ndarray::*;

use crate::internal::*;
use crate::pulse::delay::Delay;
use std::ops::Range;

/// Concat: high level concat op
#[derive(Debug, Clone, new)]
pub struct Concat {
    axis: i64,
}

impl Concat {
    fn resolve_axis(&self, rank: i64) -> TractResult<usize> {
        if 0 <= self.axis && self.axis <= rank - 1 {
            Ok(self.axis as usize)
        } else if -rank <= self.axis && self.axis < 0 {
            Ok((self.axis + rank) as usize)
        } else {
            bail!("Illegal combination of values for rank and axis: {} and {}", rank, self.axis)
        }
    }

    /// Evaluates the operation given the input tensors.
    fn eval_t<T: Datum + Copy>(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let axis = self.resolve_axis(inputs[0].shape().len() as i64)?;
        let mut slices: TVec<FixedConcatSlice<T>> = tvec![];
        for input in &inputs {
            let shape = Tensor::shape(&input);
            slices.push(FixedConcatSlice::Var(TVec::from_slice(shape)));
        }
        FixedConcat::new(axis, slices).eval(inputs)
    }
}

impl Op for Concat {
    fn name(&self) -> Cow<str> {
        "Concat".into()
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let inputs = model.node_input_facts(node.id)?;

        if let Some(super_type) = DatumType::super_type_for(inputs.iter().map(|x| x.datum_type)) {
            let axis = self.resolve_axis(inputs[0].shape.rank() as i64)?;

            fn fixed<T: Datum + Copy>(
                axis: usize,
                inputs: &[&TypedTensorInfo],
            ) -> TractResult<Box<Op>> {
                let mut slices: TVec<NormConcatSlice<T>> = tvec![];
                for input in inputs.iter() {
                    match input.konst.as_ref() {
                        Some(c_input) => {
                            slices.push(NormConcatSlice::Const(
                                c_input.cast_to::<T>()?.into_owned().into_array()?,
                            ));
                        }
                        None => {
                            slices.push(NormConcatSlice::Var(input.shape.clone()));
                        }
                    }
                }
                Ok(Box::new(NormConcat::<T>::new(axis, slices)))
            }

            let op = dispatch_copy!(fixed(super_type)(axis, &*inputs))?;
            let mut patch = TypedModelPatch::default();
            let node_id = patch.add_node(&*node.name, op, tvec!(node.outputs[0].fact.clone()))?;
            let mut inlet_slot = 0;
            for (ix, input) in inputs.iter().enumerate() {
                if input.konst.is_none() {
                    let tap = patch.tap_model(model, node.inputs[ix])?;
                    patch.add_edge(tap, InletId::new(node_id, inlet_slot))?;
                    inlet_slot += 1;
                }
            }
            patch.shunt_outside(OutletId::new(node.id, 0), OutletId::new(node_id, 0))?;
            return Ok(Some(patch));
        }
        Ok(None)
    }
}

impl StatelessOp for Concat {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let super_type: DatumType =
            DatumType::super_type_for(inputs.iter().map(|x| x.datum_type()))
                .ok_or_else(|| format!("No supertype found"))?;
        dispatch_copy!(Self::eval_t(super_type)(self, inputs))
    }
}

impl InferenceRulesOp for Concat {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        trace!("{:?}", self);
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
        let n = inputs.len() as usize;
        s.equals_all((0..n).map(|i| (&inputs[i].rank).bex()).collect())?;
        s.given(&inputs[0].rank, move |s, rank| {
            let axis = self.resolve_axis(rank as i64)?;
            s.equals(
                crate::analyser::rules::expr::SumExp::new(
                    (0..n).map(|i| (&inputs[i].shape[axis]).bex()).collect(),
                ),
                &outputs[0].shape[axis],
            )?;
            for axis in 0..axis {
                s.equals(&outputs[0].shape[axis], &inputs[0].shape[axis])?;
                s.equals_all((0..n).map(|i| inputs[i].shape[axis].bex()).collect())?;
            }
            for axis in (axis + 1)..(rank as usize) {
                s.equals(&outputs[0].shape[axis], &inputs[0].shape[axis])?;
                s.equals_all((0..n).map(|i| inputs[i].shape[axis].bex()).collect())?;
            }
            Ok(())
        })?;
        Ok(())
    }
}

/// NormConcatSlice: fully decluttered Concat equivalent
#[derive(Debug, Clone)]
pub enum NormConcatSlice<T> {
    Const(ArrayD<T>),
    Var(ShapeInfo),
}

impl<T> NormConcatSlice<T> {
    pub fn as_const(&self) -> Option<ArrayViewD<T>> {
        match self {
            NormConcatSlice::Const(c) => Some(c.view()),
            NormConcatSlice::Var(_) => None,
        }
    }
    pub fn is_var(&self) -> bool {
        match self {
            NormConcatSlice::Const(_) => false,
            NormConcatSlice::Var(_) => true,
        }
    }
}

#[derive(new, Debug, Clone)]
pub struct NormConcat<T> {
    axis: usize,
    slices: TVec<NormConcatSlice<T>>,
}

impl<T: Datum + Copy> Op for NormConcat<T> {
    fn name(&self) -> Cow<str> {
        format!("NormConcat<{:?}>", T::datum_type()).into()
    }

    fn pulsify(
        &self,
        source: &NormalizedModel,
        node: &NormalizedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
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

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let inputs = model.node_input_facts(node.id)?;

        trace!("  Entering codegen for NormConcat");

        if inputs.iter().any(|i| i.shape.as_finite().is_none()) {
            return Ok(None);
        }

        trace!("  Input has concrete finite shape");
        let shapes: TVec<TVec<usize>> =
            inputs.iter().map(|x| x.shape.as_finite().unwrap().into()).collect();

        let mut fixed_slices: TVec<FixedConcatSlice<T>> = tvec![];

        let mut input_idx = 0;
        for slice in &self.slices {
            match slice {
                NormConcatSlice::Const(c) => fixed_slices.push(FixedConcatSlice::Const(c.clone())),
                NormConcatSlice::Var(shape) => {
                    if &inputs[input_idx].shape != shape {
                        bail!("Incompatible shapes {:?} and {:?}", &inputs[input_idx].shape, shape)
                    }
                    fixed_slices.push(FixedConcatSlice::Var(shapes[input_idx].clone()));
                    input_idx += 1;
                }
            }
        }

        let op: Box<Op> = Box::new(FixedConcat::new(self.axis, fixed_slices));

        let mut patch = TypedModelPatch::default();
        let node_id = patch.add_node(&*node.name, op, tvec!(node.outputs[0].fact.clone()))?;
        for (ix, _input) in inputs.iter().enumerate() {
            let tap = patch.tap_model(model, node.inputs[ix])?;
            patch.add_edge(tap, InletId::new(node_id, ix))?;
        }
        patch.shunt_outside(OutletId::new(node.id, 0), OutletId::new(node_id, 0))?;
        return Ok(Some(patch));
    }
}

impl<T: Datum + Copy> StatelessOp for NormConcat<T> {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let casted_inputs: TVec<Cow<Tensor>> =
            inputs.iter().map(|x| x.cast_to::<T>()).collect::<TractResult<_>>()?;
        let mut mats: TVec<ArrayViewD<T>> = tvec![];
        let mut input_idx = 0;
        for slice in &self.slices {
            match slice {
                NormConcatSlice::Const(c) => mats.push(c.view()),
                NormConcatSlice::Var(shape) => {
                    let inp_view = casted_inputs[input_idx].to_array_view::<T>()?;
                    for (i_dim, e_dim) in inp_view.shape().iter().zip(shape.iter()) {
                        if let Ok(e_dim) = e_dim.to_integer() {
                            if e_dim != *i_dim as i32 {
                                bail!(
                                    "Unexpected input shape. Expected {:?}, found {:?}",
                                    shape,
                                    inp_view.shape()
                                );
                            }
                        }
                    }
                    mats.push(inp_view);
                    input_idx += 1
                }
            }
        }
        if input_idx != inputs.len() {
            bail!(
                "Unexpected number of variable inputs to NormConcat. Expected {}, got {}",
                input_idx,
                inputs.len()
            );
        }

        let result = ::ndarray::stack(Axis(self.axis), &mats)?;
        Ok(tvec![result.into_arc_tensor()])
    }
}

impl<T: Datum + Copy> NormConcat<T> {
    fn pulsify_along_concat_axis(
        &self,
        _source: &NormalizedModel,
        node: &NormalizedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        if node.inputs.len() > 1 {
            bail!("Concat can not pulse more than on input on concat axis")
        }
        let input = mapping[&node.inputs[0]];
        let mut fact = target.outlet_fact(input)?.clone();
        assert_eq!(fact.axis, self.axis);
        let var_index = self.slices.iter().position(|s| s.is_var()).unwrap();
        let pre_views: Vec<_> =
            self.slices[0..var_index].iter().map(|s| s.as_const().unwrap()).collect();
        let pre = ::ndarray::stack(Axis(self.axis), &*pre_views)?;
        let post_views: Vec<_> =
            self.slices[var_index + 1..].iter().map(|s| s.as_const().unwrap()).collect();
        let post = ::ndarray::stack(Axis(self.axis), &*post_views)?;

        let mut prec = input;
        let before = pre.shape()[self.axis];
        let after = post.shape()[self.axis];
        if fact.delay < before {
            let buffer_op = Delay::new(fact.clone(), before - fact.delay, 0);
            fact.delay = before;
            let id = target.chain_after(
                prec,
                format!("{}/Delay", node.name),
                buffer_op,
                tvec!(fact.clone()),
            )?;
            prec = OutletId::new(id, 0);
        }
        let main_op = PulsedSameAxisConcat::new(self.axis, pre, post, fact.delay, fact.dim);
        fact.dim += (before + after).to_dim();
        fact.delay -= before;
        let id = target.chain_after(prec, &*node.name, main_op, tvec!(fact))?;
        return Ok(tvec!(OutletId::new(id, 0)));
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

pub fn overwrite_part_of_pulse<T: Datum + Copy>(
    axis: usize,
    pulse_data: &mut ArrayViewMutD<T>,
    current_pos: usize,
    const_data: &ArrayViewD<T>,
    const_offset: usize,
) {
    let pulse = pulse_data.shape()[axis];
    let const_length = const_data.shape()[axis];
    let const_range = const_offset..const_offset + const_length;
    let pulse_range = current_pos..current_pos + pulse;
    let axis = Axis(axis);

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
                .assign(const_data);
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
}

/// Concat with pulse along concat axis
#[derive(new, Debug, Clone)]
pub struct PulsedSameAxisConcat<T: Datum + Copy> {
    axis: usize,
    pre_slice: ArrayD<T>,
    post_slice: ArrayD<T>,
    input_delay: usize,
    input_len: TDim,
}

impl<T: Datum + Copy> Op for PulsedSameAxisConcat<T> {
    fn name(&self) -> Cow<str> {
        format!("PulsedSameAxisConcat<{:?}>", T::datum_type()).into()
    }
}

impl<T: Datum + Copy> StatefullOp for PulsedSameAxisConcat<T> {
    fn state(&self, _session: &mut SessionState) -> TractResult<Option<Box<OpState>>> {
        return Ok(Some(Box::new(PulsedSameAxisConcatState::<T>::default())));
    }
}

impl<T: Datum + Copy> InferenceRulesOp for PulsedSameAxisConcat<T> {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        _s: &mut Solver<'r>,
        _inputs: &'p [TensorProxy],
        _outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        unreachable!();
    }
}

#[derive(Clone, Debug, Default)]
pub struct PulsedSameAxisConcatState<T: Datum + Copy> {
    current_pos: usize,
    _casper: PhantomData<T>,
}

impl<T: Datum + Copy> OpState for PulsedSameAxisConcatState<T> {
    fn eval(
        &mut self,
        session: &mut SessionState,
        op: &Op,
        mut inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        let op = op.downcast_ref::<PulsedSameAxisConcat<T>>().ok_or("Wrong Op type")?;
        let input = args_1!(inputs);
        let mut data = input.into_tensor().into_array::<T>()?;
        let pulse = data.shape()[op.axis];
        let current_pos = self.current_pos;
        self.current_pos += pulse;

        let pre_length = op.pre_slice.shape()[op.axis];
        let pre_offset = op.input_delay - pre_length;
        overwrite_part_of_pulse(
            op.axis,
            &mut data.view_mut(),
            current_pos,
            &op.pre_slice.view(),
            pre_offset,
        );
        if let Some(l) = session.known_stream_len {
            let input_length = op.input_len.eval(l as i32).unwrap() as usize;
            let post_offset = op.input_delay + input_length;
            overwrite_part_of_pulse(
                op.axis,
                &mut data.view_mut(),
                current_pos,
                &op.post_slice.view(),
                post_offset,
            );
        }

        return Ok(tvec!(data.into_arc_tensor()));
    }
}

////////////////////////////////////////////////

#[derive(new, Debug, Clone)]
pub struct FixedConcat<T> {
    axis: usize,
    slices: TVec<FixedConcatSlice<T>>,
}

#[derive(Debug, Clone)]
pub enum FixedConcatSlice<T> {
    Const(ArrayD<T>),
    Var(TVec<usize>),
}

impl<T: Datum + Copy> StatelessOp for FixedConcat<T> {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let casted_inputs: TVec<Cow<Tensor>> =
            inputs.iter().map(|x| x.cast_to::<T>()).collect::<TractResult<_>>()?;
        let mut mats: TVec<ArrayViewD<T>> = tvec![];
        let mut input_idx = 0;
        for slice in &self.slices {
            match slice {
                FixedConcatSlice::Const(c) => mats.push(c.view()),
                FixedConcatSlice::Var(shape) => {
                    let inp_view = casted_inputs[input_idx].to_array_view::<T>()?;
                    if inp_view.shape() != shape.as_slice() {
                        bail!(
                            "Unexpected input shape. Expected {:?}, found {:?}",
                            shape,
                            inp_view.shape()
                        );
                    }
                    mats.push(inp_view);
                    input_idx += 1
                }
            }
        }
        if input_idx != inputs.len() {
            bail!(
                "Unexpected number of variable inputs to FixedConcat. Expected {}, got {}",
                input_idx,
                inputs.len()
            );
        }

        let result = ::ndarray::stack(Axis(self.axis), &mats)?;
        Ok(tvec![result.into_arc_tensor()])
    }
}

impl<T: Datum + Copy> Op for FixedConcat<T> {
    fn name(&self) -> Cow<str> {
        format!("FixedConcat<{:?}>", T::datum_type()).into()
    }
}

impl<T: Datum + Copy> InferenceRulesOp for FixedConcat<T> {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        _s: &mut Solver<'r>,
        _inputs: &'p [TensorProxy],
        _outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        unreachable!();
    }
}

impl<T: Datum + Copy> InferenceRulesOp for NormConcat<T> {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        _s: &mut Solver<'r>,
        _inputs: &'p [TensorProxy],
        _outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        unreachable!();
    }
}
