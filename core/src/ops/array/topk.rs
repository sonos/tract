use std::cmp::Ordering;

use tract_data::itertools::Itertools;
use tract_ndarray::{ArrayViewMutD, Axis, Dimension};

use crate::internal::*;

#[derive(Debug, Clone, new, Default, Hash)]
pub struct Topk {
    pub axis: usize,
    pub largest: bool,
    pub fallback_k: TDim,
}

impl Op for Topk {
    fn name(&self) -> Cow<str> {
        "Topk".into()
    }

    op_as_typed_op!();
}

impl EvalOp for Topk {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (input, k) = args_2!(inputs);
        let mut output_shape: TVec<usize> = input.shape().into();
        let k = k.cast_to_scalar::<i64>()? as usize;
        output_shape[self.axis] = k;
        let dt = input.datum_type();
        let mut output_values = Tensor::zero_dt(dt, &output_shape)?;
        let mut output_indices = Tensor::zero::<i64>(&output_shape)?;
        let mut iterating_shape = output_shape.clone();
        iterating_shape[self.axis] = 1;
        let mut output_indices_view = output_indices.to_array_view_mut::<i64>()?;
        for coords in tract_ndarray::indices(&*iterating_shape) {
            let mut coords: TVec<usize> = coords.as_array_view().as_slice().unwrap().into();
            dispatch_numbers!(Self::inner_loop_t(dt)(
                self,
                &mut coords,
                &input,
                &mut output_values,
                &mut output_indices_view,
                k
            ))?;
        }
        Ok(tvec!(output_values.into_tvalue(), output_indices.into_tvalue()))
    }
}

impl Topk {
    fn inner_loop_t<T: Datum + PartialOrd>(
        &self,
        coords: &mut [usize],
        input: &Tensor,
        output_values: &mut Tensor,
        output_indices_view: &mut ArrayViewMutD<i64>,
        k: usize,
    ) -> TractResult<()> {
        let mut output_values_view = output_values.to_array_view_mut::<T>()?;
        let mut view = input.to_array_view::<T>()?;
        for (ix, x) in coords.iter().enumerate() {
            if ix != self.axis {
                view.collapse_axis(Axis(ix), *x);
            }
        }
        for (ix, (argmax, max)) in view
            .iter()
            .cloned()
            .enumerate()
            .sorted_by(|a, b| {
                let ord = { a.1.partial_cmp(&b.1).unwrap_or(Ordering::Less) };
                if self.largest {
                    ord.reverse()
                } else {
                    ord
                }
            })
            .take(k)
            .enumerate()
        {
            coords[self.axis] = ix;
            output_values_view[&*coords] = max;
            output_indices_view[&*coords] = argmax as i64;
        }
        Ok(())
    }
}

impl TypedOp for Topk {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact_values = inputs[0].without_value();
        let mut fact_indices = inputs[0].without_value();
        let k: TDim = if let Some(k) = &inputs[1].konst {
            k.cast_to_scalar::<i64>()?.into()
        } else {
            self.fallback_k.clone()
        };
        fact_values.shape.set(self.axis, k.clone());
        fact_indices.shape.set(self.axis, k);
        fact_indices.datum_type = i64::datum_type();
        Ok(tvec!(fact_values, fact_indices))
    }

    as_op!();
}
