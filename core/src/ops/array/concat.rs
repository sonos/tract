use ndarray::*;

use crate::ops::prelude::*;

#[derive(Debug, Clone, new)]
pub struct Concat {
    axis: i64,
}

#[derive(Debug, Clone)]
pub enum FixedConcatSlice<T> {
    Const(ArrayD<T>),
    Var(TVec<usize>),
}

#[derive(Debug, Clone)]
pub enum NormConcatSlice<T> {
    Const(ArrayD<T>),
    Var(TVec<TDim>),
}

impl<T> FixedConcatSlice<T> {
    pub fn shape(&self) -> &[usize] {
        match self {
            FixedConcatSlice::Const(c) => c.shape(),
            FixedConcatSlice::Var(shape) => &shape,
        }
    }
}

#[derive(new, Debug, Clone)]
pub struct FixedConcat<T> {
    axis: usize,
    slices: TVec<FixedConcatSlice<T>>,
}

#[derive(new, Debug, Clone)]
pub struct NormConcat<T> {
    axis: usize,
    slices: TVec<NormConcatSlice<T>>,
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
    fn eval_t<T: Datum + Copy>(
        &self,
        inputs: TVec<SharedTensor>,
    ) -> TractResult<TVec<SharedTensor>> {
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

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let mut inputs = model.node_input_facts(node.id)?;

        for input in inputs.iter() {
            if input.shape.as_finite().is_none() {
                return Ok(None);
            }
        }

        if let Some(super_type) = DatumType::super_type_for(inputs.iter().map(|x| x.datum_type)) {

            let axis = self.resolve_axis(inputs[0].shape.rank() as i64)?;

            fn fixed<T: Datum + Copy>(
                axis: usize,
                inputs: &[&TypedTensorInfo],
            ) -> TractResult<Box<Op>> {
                let mut slices: TVec<FixedConcatSlice<T>> = tvec![];
                for input in inputs.iter() {
                    match input.konst.as_ref() {
                        Some(c_input) => {
                            slices.push(NormConcatSlice::Const(
                                c_input.cast_to::<T>()?.into_owned().into_array()?,
                            ));
                        }
                        None => {
                            slices.push(FixedConcatSlice::Var(input.shape.as_finite().unwrap().into()));
                        }
                    }
                }
                Ok(Box::new(FixedConcat::<T>::new(axis, slices)))
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
            return Ok(Some(patch))
        }
        Ok(None)
    }
}

pub struct PulsedNormAxisConcat<T> {
    axis: usize,
    pre_slice: ArrayD<T>,
    post_slice: ArrayD<T>,
    delay: usize,
}

#[derive(Clone, Debug)]
pub struct PulsedNormAxisConcatState {
    counter: usize
}

impl OpState for PulsedNormAxisConcatState {
    fn eval(&mut self, op: &Op, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        unimplemented!()
    }
}

impl<T: Datum + Copy> StatefullOp for PulsedNormAxisConcat<T> {
    fn state(&self) -> TractResult<Option<Box<OpState>>> {
        unimplemented!()
    }
}

impl<T: Datum + Copy> Op for NormConcat<T> {
    fn name(&self) -> Cow<str> {
        format!("NormConcat<{:?}>", T::datum_type()).into()
    }

    fn pulsify(&self, inputs: TVec<&PulsedTensorFact>) -> TractResult<Vec<PulsifiedOp>> {

        if inputs.len() > 1 {
            bail!("Pulsification not implemented for more than one input to Concat")
        }

        let mut fact = inputs[0].clone();

        if fact.axis == self.axis {
            let mut input_seen = false;
            for slice in &self.slices {
                match slice {
                    NormConcatSlice::Const(c) => {
                        fact.dim += TDim::from(c.shape()[fact.axis]);
                        if !input_seen {
                            fact.delay -= c.shape()[fact.axis];
                        }
                    },
                    NormConcatSlice::Var(_) => {input_seen = true;}
                }
            }
            unimplemented!()
        } else {
            for slice in &self.slices {
                if let NormConcatSlice::Const(c) = slice {
                    fact.shape[self.axis] += c.shape()[self.axis];
                }
        }
            return Ok(vec![PulsifiedOp::new(Box::new(self.clone()), tvec![fact])])

        }
    }

    // Reduce to FixedConcat<T>
    fn reduce(
        &self,
        inputs: TVec<&TensorFact>,
        _outputs: TVec<&TensorFact>,
        phase: ReductionPhase,
    ) -> TractResult<Option<ReducedOpRewire>> {
        if phase != ReductionPhase::Codegen {
            return Ok(None);
        }

        trace!("  Entering reducer for NormConcat");

        for input in inputs.iter() {
            if input.shape.as_concrete_finite()?.is_none() {
                return Ok(None);
            }
        }

        trace!("  Input has concrete finite shape");
        let shapes: TVec<TVec<usize>> =
            inputs.iter().map(|x| x.shape.as_concrete_finite().unwrap().unwrap()).collect();

        let mut fixed_slices: TVec<FixedConcatSlice<T>> = tvec![];

        let mut input_idx = 0;
        for slice in &self.slices {
            match slice {
                NormConcatSlice::Const(c) => fixed_slices.push(FixedConcatSlice::Const(c.clone())),
                NormConcatSlice::Var(shape) => {
                    if &inputs[input_idx].shape.concretize().unwrap() != shape {
                        bail!(
                            "Incompatible shapes {:?} and {:?}",
                            &inputs[input_idx].shape.concretize().unwrap(),
                            shape
                        )
                    }
                    fixed_slices.push(FixedConcatSlice::Var(shapes[input_idx].clone()));
                    input_idx += 1;
                }
            }
        }

        Ok(Some(ReducedOpRewire::new(
            vec![Box::new(FixedConcat::new(self.axis, fixed_slices))],
            (0..inputs.len()).collect(),
        )))
    }
}

impl StatelessOp for Concat {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let super_type: DatumType =
            DatumType::super_type_for(inputs.iter().map(|x| x.datum_type()))
                .ok_or_else(|| format!("No supertype found"))?;
        dispatch_copy!(Self::eval_t(super_type)(self, inputs))
    }
}

impl<T: Datum + Copy> StatelessOp for NormConcat<T> {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let casted_inputs: TVec<Cow<Tensor>> =
            inputs.iter().map(|x| x.cast_to::<T>()).collect::<TractResult<_>>()?;
        let mut mats: TVec<ArrayViewD<T>> = tvec![];
        let mut input_idx = 0;
        for slice in &self.slices {
            match slice {
                NormConcatSlice::Const(c) => mats.push(c.view()),
                NormConcatSlice::Var(shape) => {
                    let inp_view = casted_inputs[input_idx].to_array_view::<T>()?;
                    if &inp_view.shape().iter().map(|x| TDim::from(x)).collect::<TVec<_>>() != shape
                    {
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
                "Unexpected number of variable inputs to NormConcat. Expected {}, got {}",
                input_idx,
                inputs.len()
            );
        }

        let result = ::ndarray::stack(Axis(self.axis), &mats)?;
        Ok(tvec![result.into()])
    }
}

impl<T: Datum + Copy> StatelessOp for FixedConcat<T> {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
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
        Ok(tvec![result.into()])
    }
}

impl<T: Datum + Copy> Op for FixedConcat<T> {
    fn name(&self) -> Cow<str> {
        format!("FixedConcat<{:?}>", T::datum_type()).into()
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

impl<T: Datum + Copy> InferenceRulesOp for FixedConcat<T> {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
        let n = inputs.len() as usize;
        s.equals_all((0..n).map(|i| (&inputs[i].rank).bex()).collect())?;
        let mut input_idx = 0;
        for slice in &self.slices {
            match slice {
                FixedConcatSlice::Var(shape) => {
                    for (dim_idx, dim) in shape.iter().enumerate() {
                        s.equals(dim.to_dim(), &inputs[input_idx].shape[dim_idx])?;
                    }
                    input_idx += 1;
                }
                _ => {}
            }
        }

        let common_shape = self.slices[0].shape();
        for slice in &self.slices {
            let shape = slice.shape();
            for (dim_idx, dim) in shape.iter().enumerate() {
                if dim_idx != self.axis {
                    s.equals(dim.to_dim(), common_shape[dim_idx].to_dim())?;
                }
            }
        }

        let axis_dim = {
            let mut rep = 0;
            for slice in &self.slices {
                match slice {
                    FixedConcatSlice::Var(shape) => {
                        rep += shape[self.axis];
                    }
                    FixedConcatSlice::Const(c) => {
                        rep += c.shape()[self.axis];
                    }
                }
            }
            rep
        };
        s.equals(axis_dim.to_dim(), &outputs[0].shape[self.axis])?;

        Ok(())
    }
}

impl<T: Datum + Copy> InferenceRulesOp for NormConcat<T> {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
        let n = inputs.len() as usize;
        s.equals_all((0..n).map(|i| (&inputs[i].rank).bex()).collect())?;
        let mut input_idx = 0;
        for slice in &self.slices {
            match slice {
                NormConcatSlice::Var(shape) => {
                    for (dim_idx, dim) in shape.iter().enumerate() {
                        s.equals(dim.to_dim(), &inputs[input_idx].shape[dim_idx])?;
                    }
                    input_idx += 1;
                }
                _ => {}
            }
        }
        Ok(())
    }
}
