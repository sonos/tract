use tract_data::itertools::Itertools;
use tract_ndarray::{Axis, Dimension};

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
        let mut output_values = Tensor::zero::<f32>(&output_shape)?;
        let mut output_indices = Tensor::zero::<i64>(&output_shape)?;
        let mut iterating_shape = output_shape.clone();
        iterating_shape[self.axis] = 1;
        let mut output_values_view = output_values.to_array_view_mut::<f32>()?;
        let mut output_indices_view = output_indices.to_array_view_mut::<i64>()?;
        for coords in tract_ndarray::indices(&*iterating_shape) {
            let mut coords: TVec<usize> = coords.as_array_view().as_slice().unwrap().into();
            let mut view = input.to_array_view::<f32>()?;
            for (ix, x) in coords.iter().enumerate() {
                if ix != self.axis {
                    view.index_axis_inplace(Axis(ix), *x);
                }
            }
            for (ix, (argmax, max)) in view
                .iter()
                .cloned()
                .map(|x| if self.largest { -x } else { x })
                .enumerate()
                .sorted_by(|a, b| a.1.total_cmp(&b.1))
                .take(k)
                .map(|(pos, val)| if self.largest { (pos, -val) } else { (pos, val) })
                .enumerate()
            {
                coords[self.axis] = ix;
                output_values_view[&*coords] = max;
                output_indices_view[&*coords] = argmax as i64;
            }
        }
        Ok(tvec!(output_values.into_tvalue(), output_indices.into_tvalue()))
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
