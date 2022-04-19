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

    as_op!();
}

impl EvalOp for Softmax {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let mut iterating_shape: TVec<usize> = input.shape().into();

        for i in 0..iterating_shape.len() {
            if self.axes.contains(&i) {
                iterating_shape[i] = 1
            }
        }

        let mut output = input.into_tensor().into_array::<f32>()?;

        for it_coords in tract_ndarray::indices(&*iterating_shape) {
            let mut view = output.view_mut();
            for ix in 0..iterating_shape.len() {
                if !self.axes.contains(&ix) {
                    view.collapse_axis(Axis(ix), it_coords[ix]);
                }
            }
            let max: f32 = *view.iter().max_by(|i, j| i.partial_cmp(j).unwrap()).unwrap(); //FIXME
            view.mapv_inplace(|x| (x - max).exp());
            let exp_sum: f32 = view.iter().sum();
            view.mapv_inplace(|x| x / exp_sum);
        }

        Ok(tvec!(output.into_arc_tensor()))
    }
}
