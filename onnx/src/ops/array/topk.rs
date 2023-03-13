use crate::model::ParsingContext;
use crate::pb::NodeProto;
use tract_hir::internal::*;
use tract_hir::prelude::tract_itertools::Itertools;
use tract_hir::tract_ndarray::{Axis, Dimension};

pub fn topk(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(-1i64);
    let largest = node.get_attr_opt("largest")?.unwrap_or(1i64) == 1;
    Ok((Box::new(Topk { axis, largest }), vec![]))
}

#[derive(Debug, Clone, new, Default, Hash)]
struct Topk {
    axis: i64,
    largest: bool,
}

impl Op for Topk {
    fn name(&self) -> Cow<str> {
        "Topk".into()
    }

    not_a_typed_op!();
}

impl EvalOp for Topk {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (input, k) = args_2!(inputs);
        let k = k.cast_to_scalar::<i64>()? as usize;
        let mut output_shape: TVec<usize> = input.shape().into();
        let axis =
            if self.axis >= 0 { self.axis } else { self.axis + input.rank() as i64 } as usize;
        output_shape[axis] = k;
        let mut output_values = Tensor::zero::<f32>(&output_shape)?;
        let mut output_indices = Tensor::zero::<i64>(&output_shape)?;
        let mut iterating_shape = output_shape.clone();
        iterating_shape[axis] = 1;
        let mut output_values_view = output_values.to_array_view_mut::<f32>()?;
        let mut output_indices_view = output_indices.to_array_view_mut::<i64>()?;
        for coords in tract_ndarray::indices(&*iterating_shape) {
            let mut coords: TVec<usize> = coords.as_array_view().as_slice().unwrap().into();
            let mut view = input.to_array_view::<f32>()?;
            for (ix, x) in coords.iter().enumerate() {
                if ix != axis {
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
                coords[axis] = ix;
                output_values_view[&*coords] = max;
                output_indices_view[&*coords] = argmax as i64;
            }
        }
        Ok(tvec!(output_values.into_tvalue(), output_indices.into_tvalue()))
    }
}

impl InferenceRulesOp for Topk {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 2)?;
        check_input_arity(outputs, 2)?;

        solver.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        solver.equals(&inputs[1].datum_type, i64::datum_type())?;
        solver.equals(&outputs[1].datum_type, i64::datum_type())?;

        solver.equals(&inputs[0].rank, &outputs[0].rank)?;
        solver.equals(&inputs[0].rank, &outputs[1].rank)?;
        solver.equals(&inputs[1].rank, 1)?;

        solver.equals(&inputs[1].shape[0], 1.to_dim())?;

        solver.given(&inputs[0].rank, move |s, rank| {
            let axis = if self.axis >= 0 { self.axis } else { self.axis + rank } as usize;
            for ix in 0..rank as usize {
                if ix != axis {
                    s.equals(&inputs[0].shape[ix], &outputs[0].shape[ix])?;
                    s.equals(&inputs[0].shape[ix], &outputs[1].shape[ix])?;
                } else {
                    s.given(&inputs[1].value[0], move |s, k| {
                        s.equals(&outputs[0].shape[ix], k.to_dim())?;
                        s.equals(&outputs[1].shape[ix], k.to_dim())?;
                        Ok(())
                    })?;
                }
            }
            Ok(())
        })
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(2)
    }

    as_op!();
}
