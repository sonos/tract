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

impl Concat {
    fn resolve_axis(&self, rank: i64) -> TractResult<usize> {
        if 0 <= self.axis && self.axis <= rank - 1 {
            Ok(self.axis as usize)
        } else if -rank <= self.axis && self.axis < 0 {
            Ok((self.axis + rank) as usize)
        } else {
            bail!(
                "Illegal combination of values for rank and axis: {} and {}",
                rank,
                self.axis
            )
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

    fn reduce(
        &self,
        inputs: TVec<&TensorFact>,
        _outputs: TVec<&TensorFact>,
        phase: ReductionPhase,
    ) -> TractResult<Option<ReducedOpRewire>> {
        if phase == ReductionPhase::Normalize {
            return Ok(None);
        }
        for input in inputs.iter() {
            if input.datum_type.concretize().is_none()
                || input.shape.as_concrete_finite()?.is_none()
            {
                return Ok(None);
            }
        }

        if let Some(super_type) =
            DatumType::super_type_for(inputs.iter().map(|x| x.datum_type.concretize().unwrap()))
        {
            let shapes: TVec<TVec<usize>> = inputs
                .iter()
                .map(|x| x.shape.as_concrete_finite().unwrap().unwrap())
                .collect();
            let axis = self.resolve_axis(shapes[0].len() as i64)?;
            fn fixed<T: Datum + Copy>(
                axis: usize,
                inputs: TVec<&TensorFact>,
                shapes: TVec<TVec<usize>>,
            ) -> TractResult<ReducedOpRewire> {
                let mut slices: TVec<FixedConcatSlice<T>> = tvec![];
                let mut rewired: TVec<usize> = tvec![];
                for (idx, input) in inputs.iter().enumerate() {
                    match input.concretize() {
                        Some(c_input) => {
                            slices.push(FixedConcatSlice::Const(
                                c_input.cast_to::<T>()?.into_owned().into_array()?,
                            ));
                        }
                        None => {
                            slices.push(FixedConcatSlice::Var(shapes[idx].clone()));
                            rewired.push(idx);
                        }
                    }
                }
                Ok(ReducedOpRewire::new(
                    vec![Box::new(FixedConcat::<T>::new(axis, slices))],
                    rewired,
                ))
            }
            return Ok(Some(dispatch_copy!(fixed(super_type)(
                axis, inputs, shapes
            ))?));
        }
        Ok(None)
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

impl<T: Datum + Copy> StatelessOp for FixedConcat<T> {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let casted_inputs: TVec<Cow<Tensor>> = inputs
            .iter()
            .map(|x| x.cast_to::<T>())
            .collect::<TractResult<_>>()?;
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
