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

    fn eval<T: Datum>(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
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

            let mut slices: TVec<NormConcatSlice> = tvec![];
            for input in inputs.iter() {
                match &input.konst {
                    Some(c_input) => {
                        slices.push(NormConcatSlice::Const(
                            c_input.cast_to_dt(super_type)?.into_owned(),
                        ));
                    }
                    None => slices.push(NormConcatSlice::Var),
                }
            }
            let op = NormConcat::new(axis, slices);

            let mut patch = TypedModelPatch::default();
            let node_id = patch.add_node(&*node.name, op, tvec!(node.outputs[0].fact.clone()))?;
            let mut inlet_slot = 0;
            for (ix, input) in inputs.iter().enumerate() {
                if input.konst.is_none() {
                    let mut tap = patch.tap_model(model, node.inputs[ix])?;
                    if model.outlet_fact(node.inputs[ix])?.datum_type != super_type {
                        let mut fact = model.outlet_fact(node.inputs[ix])?.clone();
                        fact.datum_type = super_type;
                        let cast = patch.chain(
                            format!("{}-Cast-{}", node.name, ix),
                            crate::ops::cast::Cast::new(super_type),
                            tvec!(fact),
                        )?;
                        tap = OutletId::new(cast, 0);
                    }
                    patch.add_edge(tap, InletId::new(node_id, inlet_slot))?;
                    inlet_slot += 1;
                }
            }
            patch.shunt_outside(OutletId::new(node.id, 0), OutletId::new(node_id, 0))?;
            return Ok(Some(patch));
        }
        Ok(None)
    }

    op_as_typed_op!();
}

impl StatelessOp for Concat {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let super_type: DatumType =
            DatumType::super_type_for(inputs.iter().map(|x| x.datum_type()))
                .ok_or_else(|| format!("No supertype found"))?;
        dispatch_datum!(Self::eval(super_type)(self, inputs))
    }
}

impl InferenceRulesOp for Concat {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
        let n = inputs.len() as usize;
        s.equals_all((0..n).map(|i| (&inputs[i].rank).bex()).collect())?;
        s.given_all((0..n).map(|i| (&inputs[i].datum_type).bex()), move |s, dts| {
            let super_type: DatumType =
                DatumType::super_type_for(dts).ok_or_else(|| format!("No supertype found"))?;
            s.equals(&outputs[0].datum_type, super_type)
        })?;
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

    inference_op_as_op!();
    to_typed!();
}

impl TypedOp for Concat {
    typed_op_as_op!();

    fn output_facts(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<TypedTensorInfo>> {
        let axis = self.resolve_axis(inputs[0].shape.rank() as i64)?;
        let mut shape = inputs[0].shape.clone();
        let dim = inputs.iter().map(|f| f.shape.dim(axis)).sum::<TDim>();
        shape.set_dim(axis, dim)?;
        Ok(tvec!(TypedTensorInfo::dt_shape(inputs[0].datum_type, shape)?))
    }
}

/// NormConcatSlice: fully decluttered Concat equivalent
#[derive(Debug, Clone)]
pub enum NormConcatSlice {
    Const(Tensor),
    Var,
}

impl NormConcatSlice {
    pub fn as_const(&self) -> Option<&Tensor> {
        match self {
            NormConcatSlice::Const(c) => Some(&c),
            NormConcatSlice::Var => None,
        }
    }

    pub fn is_var(&self) -> bool {
        match self {
            NormConcatSlice::Const(_) => false,
            NormConcatSlice::Var => true,
        }
    }
}

#[derive(new, Debug, Clone)]
pub struct NormConcat {
    pub axis: usize,
    pub slices: TVec<NormConcatSlice>,
}

impl NormConcat {
    fn to_codegen_op<T: Datum>(&self, input_shapes: &[&[usize]]) -> TractResult<Box<dyn TypedOp>> {
        let mut fixed_slices: TVec<FixedConcatSlice<T>> = tvec![];
        let mut input_idx = 0;
        for slice in &self.slices {
            match slice {
                NormConcatSlice::Const(c) => {
                    fixed_slices.push(FixedConcatSlice::Const(c.clone().into_array::<T>()?))
                }
                NormConcatSlice::Var => {
                    fixed_slices.push(FixedConcatSlice::Var(input_shapes[input_idx].into()));
                    input_idx += 1;
                }
            }
        }
        Ok(Box::new(FixedConcat::new(self.axis, fixed_slices)))
    }
}

impl Op for NormConcat {
    fn name(&self) -> Cow<str> {
        "NormConcat".into()
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let dt = model.outlet_fact(node.inputs[0])?.datum_type;
        let inputs = model.node_input_facts(node.id)?;

        trace!("  Entering codegen for NormConcat");

        if inputs.iter().any(|i| i.shape.as_finite().is_none()) {
            return Ok(None);
        }

        trace!("  Input has concrete finite shape");
        let shapes: TVec<&[usize]> = inputs.iter().map(|t| t.shape.as_finite().unwrap()).collect();

        let op = dispatch_datum!(Self::to_codegen_op(dt)(self, &*shapes))?;

        let mut patch = TypedModelPatch::default();
        let node_id = patch.add_node(&*node.name, op, tvec!(node.outputs[0].fact.clone()))?;
        for (ix, _input) in inputs.iter().enumerate() {
            let tap = patch.tap_model(model, node.inputs[ix])?;
            patch.add_edge(tap, InletId::new(node_id, ix))?;
        }
        patch.shunt_outside(OutletId::new(node.id, 0), OutletId::new(node_id, 0))?;
        return Ok(Some(patch));
    }

    op_as_typed_op!();
}

impl TypedOp for NormConcat {
    typed_op_as_op!();

    fn output_facts(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<TypedTensorInfo>> {
        let mut fact = inputs[0].clone();
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

    fn axes_info(&self, model: &TypedModel, node: &TypedNode) -> TractResult<AxesInfo> {
        if self.slices.iter().any(|s| s.as_const().is_some()) {
            Ok(AxesInfo::none())
        } else {
            let rank = model.outlet_fact(node.inputs[0])?.shape.rank();
            Ok((0..rank).filter(|&ax| ax != self.axis).map(|axis| AxisInfo::simple(axis)).collect())
        }
    }

    fn pulsify(
        &self,
        source: &NormalizedModel,
        node: &NormalizedNode,
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
            dispatch_datum!(Self::pulsify_along_concat_axis_t(fact.dt)(
                self, source, node, target, mapping
            ))
        } else {
            bail!("Pulsify for Concat on a separate axis is not implemented (but possible)");
        }
    }
}

impl StatelessOp for NormConcat {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let super_type: DatumType =
            DatumType::super_type_for(inputs.iter().map(|x| x.datum_type()))
                .ok_or_else(|| format!("No supertype found"))?;
        let shapes = inputs.iter().map(|t| t.shape()).collect::<TVec<_>>();
        let op = dispatch_datum!(Self::to_codegen_op(super_type)(self, &*shapes))?;
        std::mem::drop(shapes);
        op.as_stateless().unwrap().eval(inputs)
    }
}

impl NormConcat {
    fn pulsify_along_concat_axis_t<T: Datum>(
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
        let pre_owned = self.slices[0..var_index]
            .iter()
            .map(|s| s.as_const().unwrap().cast_to::<T>())
            .collect::<TractResult<TVec<_>>>()?;
        let pre_views =
            pre_owned.iter().map(|t| t.to_array_view::<T>()).collect::<TractResult<TVec<_>>>()?;
        let pre = T::stack_views(self.axis, &*pre_views)?;
        let post_owned = self.slices[var_index + 1..]
            .iter()
            .map(|s| s.as_const().unwrap().cast_to::<T>())
            .collect::<TractResult<TVec<_>>>()?;
        let post_views =
            post_owned.iter().map(|t| t.to_array_view::<T>()).collect::<TractResult<TVec<_>>>()?;
        let post = T::stack_views(self.axis, &*post_views)?;

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
        let main_op = PulsedSameAxisConcat::new(self.axis, pre, post, fact.delay, fact.dim.clone());
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

pub fn overwrite_part_of_pulse<T: Datum>(
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
pub struct PulsedSameAxisConcat<T: Datum> {
    axis: usize,
    pre_slice: ArrayD<T>,
    post_slice: ArrayD<T>,
    input_delay: usize,
    input_len: TDim,
}

impl<T: Datum> Op for PulsedSameAxisConcat<T> {
    fn name(&self) -> Cow<str> {
        format!("PulsedSameAxisConcat<{:?}>", T::datum_type()).into()
    }

    op_as_typed_op!();
}

impl<T: Datum> StatefullOp for PulsedSameAxisConcat<T> {
    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        return Ok(Some(Box::new(PulsedSameAxisConcatState::<T>::default())));
    }
}

impl<T: Datum> TypedOp for PulsedSameAxisConcat<T> {
    typed_op_as_op!();

    fn output_facts(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<TypedTensorInfo>> {
        Ok(tvec!(inputs[0].clone()))
    }
}

#[derive(Clone, Debug, Default)]
pub struct PulsedSameAxisConcatState<T: Datum> {
    current_pos: usize,
    _casper: PhantomData<T>,
}

impl<T: Datum> OpState for PulsedSameAxisConcatState<T> {
    fn eval(
        &mut self,
        session: &mut SessionState,
        op: &dyn Op,
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

#[derive(Debug, Clone)]
pub enum FixedConcatSlice<T> {
    Const(ArrayD<T>),
    Var(TVec<usize>),
}

impl<T> FixedConcatSlice<T> {
    pub fn as_const(&self) -> Option<ArrayViewD<T>> {
        match self {
            FixedConcatSlice::Const(a) => Some(a.view()),
            _ => None,
        }
    }
}

fn slices<'a, T: Datum>(
    slices: &'a [FixedConcatSlice<T>],
    inputs: &[&'a Tensor],
) -> TractResult<TVec<ArrayViewD<'a, T>>> {
    let mut mats: TVec<ArrayViewD<T>> = tvec![];
    let mut input_idx = 0;
    for slice in slices {
        match slice {
            FixedConcatSlice::Const(c) => mats.push(c.view()),
            FixedConcatSlice::Var(shape) => {
                let inp_view = inputs[input_idx].to_array_view::<T>()?;
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
    Ok(mats)
}

#[derive(new, Debug, Clone)]
pub struct FixedConcat<T: Datum> {
    axis: usize,
    slices: TVec<FixedConcatSlice<T>>,
}

impl<T: Datum> Op for FixedConcat<T> {
    fn name(&self) -> Cow<str> {
        format!("FixedConcat<{:?}>", T::datum_type()).into()
    }

    op_as_typed_op!();
}

impl<T: Datum> StatelessOp for FixedConcat<T> {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let casted_inputs: TVec<Cow<Tensor>> =
            inputs.iter().map(|x| x.cast_to::<T>()).collect::<TractResult<_>>()?;
        let refs: TVec<&Tensor> = casted_inputs.iter().map(|i| i.as_ref()).collect();
        let mats = slices(&self.slices, &refs)?;
        let result = T::stack_views(self.axis, &mats)?;
        Ok(tvec![result.into_arc_tensor()])
    }
}

impl<T: Datum> TypedOp for FixedConcat<T> {
    typed_op_as_op!();

    fn output_facts(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<TypedTensorInfo>> {
        let mut fact = inputs[0].clone();
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
}
