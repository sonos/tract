use std::marker::PhantomData;

use tfdeploy::analyser::rules::prelude::*;
use tfdeploy::ops::prelude::*;
use tfdeploy::TfdResult;

#[derive(Debug, Clone, Default, new)]
pub struct Fill<T: Datum> {
    _phantom: PhantomData<T>,
}

pub fn fill(pb: &::tfpb::node_def::NodeDef) -> TfdResult<Box<Op>> {
    let dtype = pb.get_attr_datum_type("T")?;
    Ok(boxed_new!(Fill(dtype)()))
}

impl<T> Op for Fill<T>
where
    T: Datum,
{
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        let (shape, value) = args_2!(inputs);
        let value = value.to_array_view()?;
        let value: T = value[[]];
        let shape = shape.to_array_view::<i32>()?;
        let array = ::ndarray::Array::from_elem(
            shape.iter().map(|i| *i as usize).collect::<Vec<usize>>(),
            value,
        );
        Ok(tvec![array.into()])
    }
}

impl<T: Datum> InferenceRulesOp for Fill<T> {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) {
        solver
            .equals(&inputs.len, 2)
            .equals(&outputs.len, 1)
            .equals(&outputs[0].datum_type, T::datum_type())
            .equals(&inputs[0].rank, 1)
            .equals(&inputs[1].rank, 0)
            .equals(outputs[0].rank.bex().to_dim(), &inputs[0].shape[0])
            .given(&outputs[0].rank, move |solver, rank: isize| {
                for dim in 0..(rank as usize) {
                    solver.equals(&outputs[0].shape[dim], inputs[0].value[dim].bex().to_dim());
                }
            });
    }
}
