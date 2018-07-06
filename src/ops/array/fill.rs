use std::collections::HashMap;
use std::marker::PhantomData;

use ops::{Attr, Op, TensorView};
use analyser::helpers::infer_forward_concrete;
use analyser::TensorFact;
use tensor::Datum;
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
    fn eval(&self, mut inputs: Vec<TensorView>) -> Result<Vec<TensorView>> {
        let (shape, value) = args_2!(inputs);
        let value = T::tensor_to_view(&value)?;
        let value = value[[]];
        let shape = i32::tensor_to_view(&shape)?;
        let array = ::ndarray::Array::from_elem(shape.iter().map(|i| *i as usize).collect::<Vec<usize>>(), value);
        Ok(vec![T::array_into_tensor(array).into()])
    }

    /// Returns the attributes of the operation and their values.
    fn get_attributes(&self) -> HashMap<&'static str, Attr> {
        hashmap!{
            "T"    => Attr::DataType(T::datatype()),
        }
    }

    /// Infers properties about the output tensors from the input tensors.
    fn infer_forward(&self, inputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>> {
        if inputs.len() < 2 {
            bail!("Fill operation needs at least one input.");
        }

        if let Some(output) = infer_forward_concrete(self, &inputs)? {
            return Ok(Some(output));
        }

        Ok(Some(vec![TensorFact::default()]))
    }

    /// Infers properties about the input tensors from the output tensors.
    fn infer_backward(&self, outputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>> {
        if outputs.len() < 1 {
            bail!("Pack operation only supports one output.");
        }

        Ok(Some(vec![TensorFact::default()]))
    }
}
