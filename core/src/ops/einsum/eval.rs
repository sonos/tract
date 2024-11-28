use super::AxesMapping;
use crate::internal::*;
use tract_data::itertools::Itertools;
use tract_linalg::Scaler;
use tract_ndarray::{Axis, Dimension};
use tract_num_traits::{One, Zero};

pub fn output_shape<D: DimLike>(
    expr: &AxesMapping,
    inputs: &[impl AsRef<[D]>],
) -> TractResult<TVec<D>> {
    Ok(expr
        .iter_all_axes()
        .filter(|a| a.outputs[0].len() > 0)
        .sorted_by_key(|axis| axis.outputs[0][0])
        .map(|axis| {
            axis.inputs[0..inputs.len()]
                .iter()
                .enumerate()
                .flat_map(|(input_id, positions)| {
                    positions.iter().map(move |p| inputs[input_id].as_ref()[*p].clone())
                })
                .find(|x| x != &1.into())
                .unwrap_or_else(|| 1.into())
        })
        .collect())
}

pub fn eval_t<Acc: Datum + Zero + One>(
    expr: &AxesMapping,
    inputs: TVec<TValue>,
) -> TractResult<Tensor> {
    let shapes: TVec<_> = inputs
        .iter()
        .map(|t| {
            if t.datum_type() == Opaque::datum_type() {
                bail!("Unoptimized einsum execution with BlockQuantized input is not implemented.");
            } else {
                Ok(t.shape())
            }
        })
        .collect::<TractResult<_>>()?;
    let output_shape = output_shape(expr, &shapes)?;
    let inputs: TVec<Cow<Tensor>> =
        inputs.iter().map(|t| t.cast_to::<Acc>()).collect::<TractResult<_>>()?;
    let inputs: TVec<tract_ndarray::ArrayViewD<Acc>> =
        inputs.iter().map(|t| t.to_array_view::<Acc>()).collect::<TractResult<_>>()?;
    let summing_axes: TVec<_> = expr
        .iter_all_axes()
        .filter(|a| {
            a.outputs[0].len() == 0 && a.inputs[0..inputs.len()].iter().any(|i| i.len() > 0)
        })
        .collect();
    let summing_shape: TVec<usize> = summing_axes
        .iter()
        .map(|axis| {
            axis.inputs
                .iter()
                .take(inputs.len())
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
        for (axis, x) in expr
            .iter_all_axes()
            .filter(|a| a.outputs[0].len() > 0)
            .sorted_by_key(|axis| axis.outputs[0][0])
            .zip(coords)
        {
            for (input_id, input_axis_positions) in axis.inputs[0..inputs.len()].iter().enumerate()
            {
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
            for (axis, x) in summing_axes.iter().zip(sum_coords) {
                for (input_id, input_axis_positions) in
                    axis.inputs.iter().take(inputs.len()).enumerate()
                {
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
    Ok(output.into_tensor())
}

pub fn eval_q(expr: &AxesMapping, qp: DatumType, inputs: TVec<TValue>) -> TractResult<Tensor> {
    let [a, b, bias, a0, a_scale, b0, b_scale, c0, c_scale] = &*inputs else {
        bail!("Expect exactly 9 inputs")
    };

    let mut a = a.cast_to::<i32>()?.into_owned();
    let a0 = a0.cast_to_scalar::<i32>()?;
    a.as_slice_mut::<i32>()?.iter_mut().for_each(|x| *x -= a0);
    let mut b = b.cast_to::<i32>()?.into_owned();
    let b0 = b0.cast_to_scalar::<i32>()?;
    b.as_slice_mut::<i32>()?.iter_mut().for_each(|x| *x -= b0);

    let mut output =
        eval_t::<i32>(expr, tvec!(a.into_tvalue(), b.into_tvalue()))?.into_array::<i32>()?;
    let scale = a_scale.cast_to_scalar::<f32>()? * b_scale.cast_to_scalar::<f32>()?
        / c_scale.cast_to_scalar::<f32>()?;
    let scale = Scaler::new(scale, tract_linalg::mmm::RoundingPolicy::Even);
    let c0 = c0.cast_to_scalar::<i32>()?;

    if bias.rank() == 0 {
        output += inputs[2].cast_to_scalar::<i32>()?;
    } else {
        let mut bias_shape = tvec!(1; output.ndim());
        bias_shape[expr.axis((InOut::In(2), 0))?.outputs[0][0]] = bias.len();
        let bias = bias.to_array_view::<i32>()?.into_shape_with_order(&*bias_shape)?;
        output = output + bias;
    }

    output.mapv_inplace(|x| x * scale);
    output.mapv_inplace(|x| x + c0);

    if qp.unquantized() == i8::datum_type() {
        output.mapv_inplace(|x| x.clamp(i8::MIN as _, i8::MAX as _))
    } else if qp.unquantized() == u8::datum_type() {
        output.mapv_inplace(|x| x.clamp(u8::MIN as _, u8::MAX as _))
    }
    Ok(output.into_tensor().cast_to_dt(qp)?.into_owned())
}
