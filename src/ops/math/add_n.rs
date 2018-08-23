use std::collections::HashMap;
use std::marker::PhantomData;

use analyser::interface::*;
use ops::prelude::*;
use tensor::Datum;
use Result;

#[derive(Debug, Clone, Default, new)]
pub struct AddN<T: Datum> {
    n: usize,
    _phantom: PhantomData<T>,
}

pub fn add_n(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
    let dtype = pb.get_attr_datum_type("T")?;
    let n = pb.get_attr_int("N")?;
    Ok(boxed_new!(AddN(dtype)(n)))
}

impl<T> Op for AddN<T>
where
    T: Datum,
{
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Value>) -> Result<TVec<Value>> {
        if inputs.len() != self.n || self.n == 0 {
            bail!("Expected {} inputs", self.n);
        }
        let mut result = inputs.pop().unwrap().into_array::<T>()?; // checked, non empty
        for input in &inputs[0..] {
            result += &input.to_array_view()?;
        }
        Ok(tvec![result.into()])
    }

    /// Returns the attributes of the operation and their values.
    fn get_attributes(&self) -> HashMap<&'static str, Attr> {
        hashmap!{
            "T"    => Attr::DatumType(T::datum_type()),
            "N"    => Attr::Usize(self.n),
        }
    }

    /// Returns a new streaming buffer for the operation.
    fn new_buffer(&self) -> Box<OpBuffer> {
        Box::new(QueuesBuffer::new(self.n))
    }

    fn step(
        &self,
        inputs: TVec<StepValue>,
        buffer: &mut Box<OpBuffer>,
    ) -> Result<Option<TVec<Value>>> {
        let buffer = buffer
            .downcast_mut::<QueuesBuffer>()
            .ok_or("The buffer can't be downcasted to QueuesBuffer.")?;

        buffer.append(inputs)?;

        if buffer.iter().any(|q| q.is_empty()) {
            Ok(None)
        } else {
            let chunks = buffer
                .iter_mut()
                .map(|b| b.pop_front().unwrap())
                .collect::<TVec<_>>();

            Ok(Some(self.eval(chunks)?))
        }
    }
}

impl<T: Datum> InferenceRulesOp for AddN<T> {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) {
        let n = self.n as isize;
        solver
            .equals(&inputs.len, n)
            .equals(&outputs.len, 1)
            .equals(&inputs[0].datum_type, &outputs[0].datum_type)
            .equals_all((0..self.n).map(|i| (&inputs[i].datum_type).bex()).collect())
            .equals(&inputs[0].rank, &outputs[0].rank)
            .equals_all((0..self.n).map(|i| inputs[i].rank.bex()).collect())
            .given(&inputs[0].rank, move |solver, rank: isize| {
                for dim in 0..(rank as usize) {
                    solver.equals(&inputs[0].shape[dim], &outputs[0].shape[dim]);
                    solver.equals_all(
                        (0..n as usize)
                            .map(|i| inputs[i].shape[dim].bex())
                            .collect(),
                    );
                }
            });
    }
}
