use std::marker::PhantomData;

use analyser::{ATensor, AShape};
use analyser::helpers::most_specific_shape;
use Result;
use super::{Input, Op};
use matrix::Datum;

#[derive(Debug, Default, new)]
pub struct Pack<T: Datum> {
    axis: usize,
    _phantom: PhantomData<T>,
}

pub fn pack(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
    let dtype = pb.get_attr_datatype("T")?;
    let axis = pb.get_attr_int("axis")?;
    Ok(boxed_new!(Pack(dtype)(axis)))
}

impl<T> Op for Pack<T>
where
    T: Datum,
{
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: Vec<Input>) -> Result<Vec<Input>> {
        use ndarray::Axis;
        let views = inputs
            .iter()
            .map(|m| {
                Ok(T::mat_to_view(&*m)?.insert_axis(Axis(self.axis)))
            })
            .collect::<Result<Vec<_>>>()?;
        let array = ::ndarray::stack(Axis(self.axis), &*views)?;
        Ok(vec![T::array_into_mat(array).into()])
    }

    /// Infers properties about the output tensors from the input tensors.
    fn infer_forward(&self, inputs: Vec<&ATensor>) -> Result<Vec<ATensor>> {
        if inputs.len() < 1 {
            bail!("Pack operation needs at least one input.");
        }

        try_infer_forward_concrete!(self, &inputs);

        // If we don't know the actual value, we can still compute the shape.
        let n = inputs.len();
        let shapes = inputs
            .iter()
            .map(|t| &t.shape);

        // We get the most specific shape, and replace the axis with an unknown.
        let mut shape = most_specific_shape(shapes)?.inner().clone();
        shape.insert(self.axis, adimension!(n));

        let output = ATensor {
            datatype: inputs[0].datatype.clone(),
            shape: AShape::Closed(shape),
            value: avalue!(_),
        };

        Ok(vec![output])
    }

    /// Infers properties about the input tensors from the output tensors.
    fn infer_backward(&self, outputs: Vec<&ATensor>) -> Result<Vec<ATensor>> {
        if outputs.len() < 1 {
            bail!("Pack operation only supports one output.");
        }

        // The operation adds a dimension, so all we have to do is remove it.
        let shape = match &outputs[0].shape {
            AShape::Open(v) => {
                let mut inner = v.clone();
                if self.axis > inner.len() {
                    inner.remove(self.axis);
                }

                AShape::Open(inner)
            },

            AShape::Closed(v) => {
                let mut inner = v.clone();
                inner.remove(self.axis);

                AShape::Closed(inner)
            }
        };

        Ok(vec![ATensor {
            datatype: outputs[0].datatype.clone(),
            shape,
            value: avalue!(_)
        }])
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use Matrix;
    use super::*;
    use ndarray::arr2;

    #[test]
    fn pack_0() {
        let inputs = vec![
            Matrix::i32s(&[2], &[1, 4]).unwrap().into(),
            Matrix::i32s(&[2], &[2, 5]).unwrap().into(),
            Matrix::i32s(&[2], &[3, 6]).unwrap().into(),
        ];
        assert_eq!(
            Pack::<i32>::new(0)
                .eval(inputs.clone())
                .unwrap()
                .remove(0)
                .into_matrix(),
            Matrix::from(arr2(&[[1, 4], [2, 5], [3, 6]]))
        );
        assert_eq!(
            Pack::<i32>::new(1)
                .eval(inputs.clone())
                .unwrap()
                .remove(0)
                .into_matrix(),
            Matrix::from(arr2(&[[1, 2, 3], [4, 5, 6]]))
        );
    }

    #[test]
    fn pack_1() {
        let pack = Pack::<i32>::new(0);
        let input = Matrix::i32s(&[0], &[]).unwrap();
        let exp: Matrix = Matrix::i32s(&[1, 0], &[]).unwrap();
        let found = pack.eval(vec![input.into()]).unwrap();

        assert!(
            exp.close_enough(&found[0]),
            "expected: {:?} found: {:?}",
            exp,
            found[0]
        )
    }
}
