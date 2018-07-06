use std::collections::HashMap;
use std::marker::PhantomData;

use ops::{Attr, Op, TensorView};
use analyser::helpers::infer_forward_concrete;
use analyser::helpers::most_specific_shape;
use analyser::{ShapeFact, TensorFact, ValueFact};
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

    /// Infers properties about the output tensors from the input tensors.
    fn infer_forward(&self, inputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>> {
        if inputs.len() < 1 {
            bail!("Pack operation needs at least one input.");
        }

        if let Some(output) = infer_forward_concrete(self, &inputs)? {
            return Ok(Some(output));
        }

        // If we don't know the actual value, we can still compute the shape.
        let shapes = inputs.iter().map(|t| &t.shape);

        // We get the most specific shape, and replace the axis with an unknown.
        let shape = most_specific_shape(shapes)?;

        let output = TensorFact {
            datatype: inputs[0].datatype,
            shape: shape.map(|s| s.clone()).unwrap_or(ShapeFact::any()),
            value: valuefact!(_),
        };

        Ok(Some(vec![output]))
    }

    /// Infers properties about the input tensors from the output tensors.
    fn infer_backward(&self, outputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>> {
        let input_fact = TensorFact {
            value: ValueFact::Any,
            .. outputs[0].clone()
        };
        let inputs = (0..self.n).map(|_| input_fact.clone()).collect();
        Ok(Some(inputs))
    }
}
