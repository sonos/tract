use super::AxesMapping;
use crate::internal::*;
use ndarray::{ArrayViewD, Zip};
use tract_data::itertools::Itertools;
use tract_linalg::block_quant::BlockQuantValue;
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

pub fn dequant_inputs(acc: DatumType, input: TVec<TValue>) -> TractResult<TVec<TValue>> {
    input.into_iter().map(|i| if i.datum_type().is_number() { Ok(i) } else {
        let bqvs = i.as_slice::<Opaque>()?.iter().map(|o| o.downcast_ref::<BlockQuantValue>()).collect::<Option<Vec<&BlockQuantValue>>>().context("Numbers and BlockQuantValues are the only supported input for unoptimized einsum")?;
        let mut unpacked:Vec<Tensor> = if acc.is::<f16>() {
             bqvs.iter().map(|bqv| bqv.fact.format.dequant_f16(&bqv.value)).collect::<TractResult<_>>()?
         } else if acc.is::<f32>() {
             bqvs.iter().map(|bqv| bqv.fact.format.dequant_f32(&bqv.value)).collect::<TractResult<_>>()?
         } else {
             bail!("Only f32 and f16 accumulators are compatible with BlockQuantValue inputs");
         }    ;
         unpacked.iter_mut().try_for_each(|t| t.insert_axis(0))?;
         let stacked = Tensor::stack_tensors(0, &unpacked)?;
         let shape = i.shape().iter().chain(bqvs[0].fact.shape()).copied().collect_vec();
         Ok(stacked.into_shape(&shape)?.into_tvalue())
    } ).collect::<TractResult<TVec<TValue>>>()
}

pub fn eval_t<Acc: Datum + Zero + One>(
    expr: &AxesMapping,
    inputs: TVec<TValue>,
) -> TractResult<Tensor> {
    let inputs = dequant_inputs(Acc::datum_type(), inputs)?;
    let shapes: TVec<_> = inputs.iter().map(|t| t.shape()).collect();
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
    fn reshape_param<'a>(
        expr: &AxesMapping,
        data_slot: InOut,
        qp: &'a Tensor,
        qp_slot: InOut,
    ) -> TractResult<ArrayViewD<'a, f32>> {
        if qp.rank() == 0 {
            qp.to_array_view()
        } else {
            let data_rank = expr.rank(data_slot);

            // Handle case where axis is not present in input (qp.len is necessarily 1)
            let pos_in_input =
                expr.axis((qp_slot, 0))?.interface(data_slot).first().cloned().unwrap_or(0);

            let mut shape = vec![1; data_rank];
            shape[pos_in_input] = qp.len();
            Ok(qp.to_array_view()?.into_shape_with_order(shape)?)
        }
    }
    let [a, b, bias, a0, a_scale, b0, b_scale, c0, c_scale] = &*inputs else {
        bail!("Expect exactly 9 inputs")
    };

    let mut a = a.cast_to::<i32>()?.cast_to::<f32>()?.into_owned();
    let mut b = b.cast_to::<i32>()?.cast_to::<f32>()?.into_owned();

    let a0 = a0.cast_to::<f32>()?;
    let b0 = b0.cast_to::<f32>()?;
    let c0 = c0.cast_to::<f32>()?;
    let a_scale = a_scale.cast_to::<f32>()?;
    let b_scale = b_scale.cast_to::<f32>()?;
    let c_scale = c_scale.cast_to::<f32>()?;
    let bias = bias.cast_to::<f32>()?;
    ensure!(a0.rank() < 2);
    ensure!(b0.rank() < 2);
    ensure!(c0.rank() < 2);
    ensure!(a_scale.rank() < 2);
    ensure!(b_scale.rank() < 2);
    ensure!(c_scale.rank() < 2);
    ensure!(bias.rank() < 2);

    Zip::from(a.to_array_view_mut::<f32>()?)
        .and_broadcast(reshape_param(expr, InOut::In(0), &a0, InOut::In(3))?)
        .and_broadcast(reshape_param(expr, InOut::In(0), &a_scale, InOut::In(4))?)
        .for_each(|a, a0, a_scale| *a = a_scale * (*a - a0));

    Zip::from(b.to_array_view_mut::<f32>()?)
        .and_broadcast(reshape_param(expr, InOut::In(1), &b0, InOut::In(5))?)
        .and_broadcast(reshape_param(expr, InOut::In(1), &b_scale, InOut::In(6))?)
        .for_each(|b, b0, b_scale| *b = b_scale * (*b - b0));

    let mut output =
        eval_t::<f32>(expr, tvec!(a.into_tvalue(), b.into_tvalue()))?.into_array::<f32>()?;

    Zip::from(&mut output)
        .and_broadcast(reshape_param(expr, InOut::Out(0), &bias, InOut::In(2))?)
        .and_broadcast(reshape_param(expr, InOut::Out(0), &c0, InOut::In(7))?)
        .and_broadcast(reshape_param(expr, InOut::Out(0), &c_scale, InOut::In(8))?)
        .and_broadcast(reshape_param(expr, InOut::Out(0), &a_scale, InOut::In(4))?)
        .and_broadcast(reshape_param(expr, InOut::Out(0), &b_scale, InOut::In(6))?)
        .for_each(|c, bias, c0, c_scale, a_scale, b_scale| {
            *c = ((*c + bias * a_scale * b_scale) / c_scale + c0).round()
        });

    if qp.unquantized() == i8::datum_type() {
        output.mapv_inplace(|x| x.clamp(i8::MIN as _, i8::MAX as _))
    } else if qp.unquantized() == u8::datum_type() {
        output.mapv_inplace(|x| x.clamp(u8::MIN as _, u8::MAX as _))
    }
    Ok(output.into_tensor().cast_to::<i32>()?.cast_to_dt(qp)?.into_owned())
}
