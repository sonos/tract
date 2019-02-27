use ndarray::*;

use crate::ops::prelude::*;

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
        let mats: Vec<Cow<Tensor>> = inputs
            .iter()
            .map(|mat| mat.cast_to::<T>())
            .collect::<TractResult<_>>()?;
        let views: Vec<_> = mats
            .iter()
            .map(|mat| mat.to_array_view::<T>())
            .collect::<TractResult<_>>()?;
        let result = ::ndarray::stack(Axis(axis), &views)?;
        Ok(tvec![result.into()])
    }
}

impl Op for Concat {
    fn name(&self) -> Cow<str> {
        "Concat".into()
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
