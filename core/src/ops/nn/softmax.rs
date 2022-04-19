use crate::internal::*;
use ndarray::prelude::*;

#[derive(Debug, Clone, new, Educe)]
#[educe(Hash)]
pub struct Softmax {
    pub axes: TVec<usize>,
    #[educe(Hash(method = "hash_f32"))]
    pub beta: f32,
}

impl_dyn_hash!(Softmax);

impl Op for Softmax {
    fn name(&self) -> Cow<str> {
        "Softmax".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("Axis: {:?}", self.axes), format!("Beta: {:?}", self.beta)])
    }

    op_core_mir!();
    op_as_typed_op!();
}

impl TypedOp for Softmax {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let fact = inputs[0].clone();
        Ok(tvec!(fact))
    }

    as_op!();
}

impl EvalOp for Softmax {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let input_dt = input.datum_type();
        let input = input.to_array_view()?;
        let mut output = input.to_owned();

        let row_max = output.fold_axis(Axis(1), 0.0, |max_value, x: &f32| x.max(*max_value));
        output.axis_iter_mut(Axis(0)).zip(row_max.iter()).for_each(|(mut row_x, max_i)| {
            row_x.iter_mut().for_each(|x| *x = (*x - max_i).exp());
        });
        let row_exp_sum = output.fold_axis(Axis(1), 0.0, |exp_sum, x| *x + exp_sum);
        output.axis_iter_mut(Axis(0)).zip(row_exp_sum.iter()).for_each(|(mut row_x, exp_sum_i)| {
            row_x.iter_mut().for_each(|x| *x = *x / exp_sum_i);
        });

        let mut output = output.into_tensor();
        unsafe { output.set_datum_type(input_dt) };

        Ok(tvec!(output.into_arc_tensor()))
    }
}
