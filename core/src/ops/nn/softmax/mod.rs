mod fixedpoint;
pub mod math;

use math::{
    exp_on_negative_values, get_reciprocal, is_signed, rescale, rounding_divide_by_pot,
    saturating_rounding_doubling_high_mul,
};
use num_traits::{AsPrimitive, Float, PrimInt};
use std::fmt::Debug;

use crate::internal::*;
use ndarray::prelude::*;

#[derive(Debug, Clone, new, Hash)]
pub struct Softmax {
    pub axes: TVec<usize>,
}

impl_dyn_hash!(Softmax);

impl Op for Softmax {
    fn name(&self) -> Cow<str> {
        "Softmax".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("Axis: {:?}", self.axes)])
    }

    op_core_mir!();
    op_as_typed_op!();
}

impl TypedOp for Softmax {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let fact = inputs[0].clone();
        Ok(tvec!(fact))
    }

    fn invariants(
        &self,
        inputs: &[&TypedFact],
        _outputs: &[&TypedFact],
    ) -> TractResult<Invariants> {
        let axes = (0..inputs[0].rank()).map(|axis| AxisInfo::simple(axis)).collect();
        Ok(axes)
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        let axes: Option<TVec<usize>> =
            self.axes.iter().map(|it| change.transform_axis(*it)).collect();
        if let Some(axes) = axes {
            Ok(Some(AxisChangeConsequence::new(
                model,
                node,
                Some(Box::new(Softmax { axes })),
                change,
            )))
        } else {
            Ok(None)
        }
    }

    as_op!();
}

impl EvalOp for Softmax {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let dt = input.datum_type();

        let output = match dt {
            DatumType::F32 => self.eval_t::<f32>(input)?,
            DatumType::QI8(_) => self.eval_quant_t::<i8>(input)?,
            DatumType::QU8(_) => self.eval_quant_t::<u8>(input)?,
            dt => bail!("Unsupported type {:?}", dt),
        };
        Ok(output)
    }
}

impl Softmax {
    fn eval_t<T>(&self, input: Arc<Tensor>) -> TractResult<TVec<Arc<Tensor>>>
    where
        T: Float + Datum + AsPrimitive<f32>,
        f32: AsPrimitive<T>
    {
        let mut iterating_shape: TVec<usize> = input.shape().into();

        for i in 0..iterating_shape.len() {
            if self.axes.contains(&i) {
                iterating_shape[i] = 1
            }
        }

        let mut output = input.into_tensor().into_array::<T>()?;

        for it_coords in tract_ndarray::indices(&*iterating_shape) {
            let mut view = output.view_mut();
            for ix in 0..iterating_shape.len() {
                if !self.axes.contains(&ix) {
                    view.collapse_axis(Axis(ix), it_coords[ix]);
                }
            }

            let max = *view.iter().max_by(|i, j| i.partial_cmp(j).unwrap()).unwrap();
            view.mapv_inplace(|x| (x - max).exp());
            let exp_sum: f32 = view.iter().map(|it| it.as_()).sum();
            view.mapv_inplace(|x| x / exp_sum.as_());
        }

        Ok(tvec!(output.into_arc_tensor()))

    }

    fn eval_quant_t<T>(&self, input: Arc<Tensor>) -> TractResult<TVec<Arc<Tensor>>>
    where
        T: PrimInt + Datum + AsPrimitive<i32> + Debug,
        i32: AsPrimitive<T>,
    {
        let fixed_point = 0; // TODO
        let mut iterating_shape: TVec<usize> = input.shape().into();

        for i in 0..iterating_shape.len() {
            if self.axes.contains(&i) {
                iterating_shape[i] = 1
            }
        }

        let mut output = input.into_tensor().into_array::<T>()?;

        for it_coords in tract_ndarray::indices(&*iterating_shape) {
            let mut view = output.view_mut();
            for ix in 0..iterating_shape.len() {
                if !self.axes.contains(&ix) {
                    view.collapse_axis(Axis(ix), it_coords[ix]);
                }
            }
            softmax_quant_inner(view, fixed_point);
        }

        Ok(tvec!(output.into_arc_tensor()))
    }
}



fn softmax_quant_inner<T, D: Dimension>(mut view: ArrayViewMut<T, D>, fixed_point: usize)
where
    T: PrimInt + AsPrimitive<i32> + Debug,
    i32: AsPrimitive<T>,
{
    assert!(view.len() > 1);
    // Compute the max
    let max = view.iter().max().unwrap();

    // Compute exponentiation element wise in a buffer
    let mut buffer = vec![0_i32; view.len()];
    let shift = {
        if is_signed::<T>() {
            26 - (std::mem::size_of::<T>() * 8 - 1 - fixed_point)
        } else {
            26 - (std::mem::size_of::<T>() * 8 - fixed_point)
        }
    };

    view
        .iter()
        .zip(buffer.iter_mut())
        .for_each(|(x, exp)| *exp = exp_on_negative_values((x.as_() - max.as_()) << shift));

    // Compute sum of exp and 1/sum of exp
    let sum_of_exp = buffer.iter().map(|it| rescale(*it, 0, 12)).sum();
    let (inv_sum_of_exp, num_bits_over_unit) = get_reciprocal(sum_of_exp, 12);

    // Do the final computation
    view.iter_mut().zip(buffer.iter()).for_each(|(it, exp)| {
        let exponent = if is_signed::<T>() {
            num_bits_over_unit + 31 - (std::mem::size_of::<T>() * 8) + 1
        } else {
            num_bits_over_unit + 31 - std::mem::size_of::<T>() * 8
        };

        let unsat_output = rounding_divide_by_pot(
            saturating_rounding_doubling_high_mul(inv_sum_of_exp, *exp),
            exponent as i32,
        );
        *it = i32::max(i32::min(unsat_output, T::max_value().as_()), T::min_value().as_()).as_()
    })
}