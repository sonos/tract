use std::collections::HashMap;
use std::marker::PhantomData;

use analyser::helpers::infer_forward_concrete;
use ndarray::{Array, ArrayD, ArrayView2, ArrayViewD};

use analyser::TensorFact;
use ops::{Attr, Op, OpBuffer, TensorView};
use tensor::Datum;
use Result;

#[derive(Debug, Clone, Default, new)]
pub struct Pad<T: Datum> {
    _phantom: PhantomData<T>,
}

pub fn pad(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
    let dtype = pb.get_attr_datatype("T")?;
    Ok(boxed_new!(Pad(dtype)()))
}

impl<T: Datum> Pad<T> {
    fn compute(
        input: &ArrayViewD<T>,
        paddings: ArrayView2<i32>,
        stream_dim: Option<usize>,
    ) -> Result<ArrayD<T>> {
        let shape: Vec<usize> = input
            .shape()
            .iter()
            .enumerate()
            .map(|(ix, &dim)| {
                if Some(ix) != stream_dim {
                    dim + (paddings[(ix, 0)] + paddings[(ix, 1)]) as usize
                } else {
                    dim
                }
            })
            .collect();
        let mut index_in_input = vec![0; input.ndim()];
        let result = Array::from_shape_fn(shape, |index| {
            for i in 0..input.ndim() {
                if index[i] < paddings[(i, 0)] as usize
                    || index[i] - paddings[(i, 0)] as usize >= input.shape()[i] as usize
                {
                    return T::zero();
                } else {
                    index_in_input[i] = index[i] - paddings[(i, 0)] as usize;
                };
            }
            input[&*index_in_input]
        });
        Ok(result)
    }
}

impl<T> Op for Pad<T>
where
    T: Datum,
{
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: Vec<TensorView>) -> Result<Vec<TensorView>> {
        let (input, paddings) = args_2!(inputs);
        let input = T::tensor_to_view(&input)?;
        let paddings = i32::tensor_to_view(&paddings)?.into_dimensionality()?;
        Ok(vec![
            T::array_into_tensor(Self::compute(&input, paddings, None)?).into(),
        ])
    }

    /// Returns the attributes of the operation and their values.
    fn get_attributes(&self) -> HashMap<&'static str, Attr> {
        hashmap!{
            "T"    => Attr::DataType(T::datatype()),
        }
    }

    /// Infers properties about the output tensors from the input tensors.
    fn infer_forward(&self, mut inputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>> {
        use analyser::*;

        if inputs.len() != 2 {
            bail!("Pad operation needs at least one input.");
        }

        // if we know everything...
        if let Some(output) = infer_forward_concrete(self, &inputs)? {
            return Ok(Some(output));
        }
        let (input_fact, padding_fact) = args_2!(inputs);

        let mut output_fact = TensorFact {
            // propagate input type to output
            datatype: input_fact.datatype,
            ..TensorFact::default()
        };

        if let (Some(mut shape), Some(pad)) = (
            input_fact.shape.concretize(),
            padding_fact.value.concretize(),
        ) {
            let pad = i32::tensor_to_view(pad)?;
            shape
                .iter_mut()
                .zip(pad.outer_iter())
                .for_each(|(s, p)| *s += p[0] as usize + p[1] as usize);
            output_fact.shape = shape.into();
        }

        Ok(Some(vec![output_fact]))
    }

    /// Infers properties about the input tensors from the output tensors.
    fn infer_backward(&self, _outputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>> {
        // FIXME
        Ok(Some(vec![TensorFact::default(), TensorFact::default()]))
    }

    fn step(
        &self,
        mut inputs: Vec<(Option<usize>, Option<TensorView>)>,
        _buffer: &mut Box<OpBuffer>,
    ) -> Result<Option<Vec<TensorView>>> {
        if let ((Some(stream_dim), Some(chunk)), (None, Some(paddings))) = args_2!(inputs) {
            let chunk = T::tensor_to_view(&chunk)?;
            let paddings = i32::tensor_to_view(&paddings)?.into_dimensionality()?;
            Ok(Some(vec![
                T::array_into_tensor(Self::compute(&chunk, paddings, Some(stream_dim))?).into(),
            ]))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    use Tensor;

    #[test]
    fn pad_0() {
        let inputs = vec![
            Tensor::from(arr2(&[[1, 2, 3], [4, 5, 6]])).into(),
            Tensor::from(arr2(&[[1, 1], [2, 2]])).into(),
        ];

        let expected = Tensor::from(arr2(&[
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 2, 3, 0, 0],
            [0, 0, 4, 5, 6, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]));

        assert_eq!(
            Pad::<i32>::new().eval(inputs).unwrap(),
            vec![expected.into()]
        );
    }
}
