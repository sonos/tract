use std::collections::HashMap;
use std::marker::PhantomData;

use ops::prelude::*;
use analyser::interface::*;
use tensor::Datum;
use Result;

#[derive(Debug, Clone, Default, new)]
pub struct AddN<T: Datum> {
    n: usize,
    _phantom: PhantomData<T>,
}

pub fn add_n(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
    let dtype = pb.get_attr_datatype("T")?;
    let n = pb.get_attr_int("N")?;
    Ok(boxed_new!(AddN(dtype)(n)))
}

impl<T> Op for AddN<T>
where
    T: Datum,
{
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: Vec<TensorView>) -> Result<Vec<TensorView>> {
        if inputs.len() != self.n || self.n == 0 {
            bail!("Expected {} inputs", self.n);
        }
        let mut result = T::tensor_into_array(inputs.pop().unwrap().into_tensor())?; // checked, non empty
        for input in &inputs[0..] {
            result += &T::tensor_to_view(input.as_tensor())?;
        }
        Ok(vec![T::array_into_tensor(result).into()])
    }

    /// Returns the attributes of the operation and their values.
    fn get_attributes(&self) -> HashMap<&'static str, Attr> {
        hashmap!{
            "T"    => Attr::DataType(T::datatype()),
            "N"    => Attr::Usize(self.n),
        }
    }
}

impl<T:Datum> InferenceRulesOp for AddN<T> {
    fn rules<'r, 'p: 'r>(&self, solver: &mut Solver<'r>, inputs: &'p TensorsProxy, outputs: &'p TensorsProxy) {

        let n = self.n as isize;
        solver
            .equals(&inputs.len, n)
            .equals(&outputs.len, 1)
            .equals(&inputs[0].datatype, &outputs[0].datatype)
            .equals_all((0..self.n).map(|i| bexp(&inputs[i].datatype)).collect())

            .equals(&inputs[0].rank, &outputs[0].rank)
            .equals_all((0..self.n).map(|i| bexp(&inputs[i].rank)).collect())

            .given(&inputs[0].rank, move |solver, rank: usize| {
                for dim in 0..rank {
                    solver.equals(&inputs[0].shape[dim], &outputs[0].shape[dim]);
                    solver.equals_all((0..n as usize).map(|i| bexp(&inputs[i].shape[dim])).collect());
                }
            })
        ;
    }
}
