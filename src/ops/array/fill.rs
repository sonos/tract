use std::collections::HashMap;
use std::marker::PhantomData;

use analyser::interface::*;
use ops::prelude::*;
use Result;

#[derive(Debug, Clone, Default, new)]
pub struct Fill<T: Datum> {
    _phantom: PhantomData<T>,
}

pub fn fill(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
    let dtype = pb.get_attr_datatype("T")?;
    Ok(boxed_new!(Fill(dtype)()))
}

impl<T> Op for Fill<T>
where
    T: Datum,
{
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: Vec<Value>) -> Result<Vec<Value>> {
        let (shape, value) = args_2!(inputs);
        let value = T::tensor_to_view(&value)?;
        let value = value[[]];
        let shape = i32::tensor_to_view(&shape)?;
        let array = ::ndarray::Array::from_elem(
            shape.iter().map(|i| *i as usize).collect::<Vec<usize>>(),
            value,
        );
        Ok(vec![T::array_into_tensor(array).into()])
    }

    /// Returns the attributes of the operation and their values.
    fn get_attributes(&self) -> HashMap<&'static str, Attr> {
        hashmap!{
            "T"    => Attr::DatumType(T::datatype()),
        }
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
            .equals(&outputs[0].datatype, T::datatype())
            .equals(&inputs[0].rank, 1)
            .equals(&inputs[1].rank, 0)
            .equals(&outputs[0].rank, &inputs[0].shape[0])
            .given(&outputs[0].rank, move |solver, rank: usize| {
                for dim in 0..rank {
                    solver.equals(&outputs[0].shape[dim], &inputs[0].value[dim]);
                }
            });
    }
}
