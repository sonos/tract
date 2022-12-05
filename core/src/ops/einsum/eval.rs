use super::Expr;
use crate::internal::*;
use tract_data::itertools::Itertools;
use tract_ndarray::{Axis, Dimension};
use tract_num_traits::{One, Zero};

pub fn output_shape<D: DimLike>(expr: &Expr, inputs: &[&[D]]) -> TVec<D> {
    expr.index
        .iter()
        .sorted_by_key(|axis| axis.result.unwrap())
        .map(|axis| {
            axis.inputs
                .iter()
                .enumerate()
                .flat_map(|(input_id, positions)| {
                    positions.iter().map(move |p| inputs[input_id][*p].clone())
                })
                .find(|x| x != &1.into())
                .unwrap_or_else(|| 1.into())
        })
        .collect()
}

pub fn eval_t<Acc: Datum + Zero + One>(
    expr: &Expr,
    inputs: TVec<TValue>,
) -> TractResult<TVec<TValue>> {
    let shapes: TVec<_> = inputs.iter().map(|t| t.shape()).collect();
    let output_shape = output_shape(expr, &shapes);
    let inputs:TVec<Cow<Tensor>> = inputs.iter().map(|t| t.cast_to::<Acc>()).collect::<TractResult<_>>()?;
    let inputs: TVec<tract_ndarray::ArrayViewD<Acc>> =
        inputs.iter().map(|t| t.to_array_view::<Acc>()).collect::<TractResult<_>>()?;
    let summing_shape: TVec<usize> = expr
        .sum
        .iter()
        .map(|axis| {
            axis.inputs
                .iter()
                .enumerate()
                .find_map(|(input_id, positions)| {
                    if positions.len() > 0 {
                        Some(inputs[input_id].shape()[positions[0]])
                    } else {
                        None
                    }
                })
                .unwrap()
        })
        .collect();
    let output = tract_ndarray::ArrayD::<Acc>::from_shape_fn(&*output_shape, |coords| {
        let coords = coords.as_array_view();
        let mut views = inputs.clone();
        for (axis, x) in expr.index.iter().sorted_by_key(|axis| axis.result.unwrap()).zip(coords) {
            for (input_id, input_axis_positions) in axis.inputs.iter().enumerate() {
                for position in input_axis_positions {
                    let x = if views[input_id].shape()[*position] == 1 { 0 } else { *x };
                    views[input_id]
                        .slice_axis_inplace(tract_ndarray::Axis(*position), (x..=x).into());
                }
            }
        }
        let mut sum: Acc = Acc::zero();
        for sum_coords in tract_ndarray::indices(&*summing_shape) {
            let mut views = views.clone();
            let sum_coords = sum_coords.as_array_view();
            for (axis, x) in expr.sum.iter().zip(&sum_coords) {
                for (input_id, input_axis_positions) in axis.inputs.iter().enumerate() {
                    for position in input_axis_positions {
                        views[input_id].slice_axis_inplace(Axis(*position), (*x..=*x).into())
                    }
                }
            }
            let mut product = Acc::one();
            for v in &views {
                debug_assert_eq!(v.len(), 1);
                product = product * v.iter().next().unwrap().clone();
            }
            sum = sum + product;
        }
        sum
    });
    Ok(tvec!(output.into_tvalue()))
}
