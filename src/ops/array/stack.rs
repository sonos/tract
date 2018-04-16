use std::marker::PhantomData;

use Result;
use super::{Input, Op};
use tfpb::types::DataType;
use matrix::Datum;

#[derive(Debug, Default, new)]
pub struct Stack<T: Datum> {
    axis: usize,
    _phantom: PhantomData<T>,
}

pub fn build(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
    let dtype: DataType = pb.get_attr()
        .get("T")
        .ok_or("Stack expect T attribute")?
        .get_field_type();
    let axis = pb.get_attr()
        .get("axis")
        .ok_or("Stack expect axis attribute")?
        .get_i() as usize;
    Ok(boxed_new!(Stack(dtype)(axis)))
}

impl<T> Op for Stack<T>
where
    T: Datum,
{
    fn eval(&self, inputs: Vec<Input>) -> Result<Vec<Input>> {
        let views = inputs
            .iter()
            .map(|m| T::mat_to_view(&*m))
            .collect::<Result<Vec<_>>>()?;
        let array = ::ndarray::stack(::ndarray::Axis(self.axis), &*views)?;
        Ok(vec![T::array_into_mat(array).into()])
    }
}

