use std::collections::HashMap;
use std::marker::PhantomData;

use ndarray::{Array, ArrayD, ArrayView2, ArrayViewD};

use analyser::interface::*;
use ops::prelude::*;
use tensor::Datum;
use Result;

#[derive(Debug, Clone, Default, new)]
pub struct Pad<T: Datum> {
    _phantom: PhantomData<T>,
}

pub fn pad(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
    let dtype = pb.get_attr_datum_type("T")?;
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
    fn eval(&self, mut inputs: TVec<Value>) -> Result<TVec<Value>> {
        let (input, paddings) = args_2!(inputs);
        let input = input.to_array_view::<T>()?;
        let paddings = paddings.to_array_view::<i32>()?.into_dimensionality()?;
        Ok(tvec![Self::compute(&input, paddings, None)?.into()])
    }

    /// Returns the attributes of the operation and their values.
    fn get_attributes(&self) -> HashMap<&'static str, Attr> {
        hashmap!{
            "T"    => Attr::DatumType(T::datum_type()),
        }
    }

    fn step(
        &self,
        mut inputs: TVec<StepValue>,
        _buffer: &mut Box<OpBuffer>,
    ) -> Result<Option<TVec<Value>>> {
        if let (StepValue::Stream(stream), StepValue::Const(paddings)) = args_2!(inputs) {
            if let Some(chunk) = stream.chunk {
                let chunk = chunk.to_array_view::<T>()?;
                let paddings = i32::tensor_to_view(&paddings)?.into_dimensionality()?;
                return Ok(Some(tvec![
                    Self::compute(&chunk, paddings, Some(stream.info.axis))?
                        .into(),
                ]));
            }
        }
        Ok(None)
    }
}

impl<T: Datum> InferenceRulesOp for Pad<T> {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) {
        let input = &inputs[0];
        let padding = &inputs[1];
        let output = &outputs[0];
        solver
            .equals(&inputs.len, 2)
            .equals(&outputs.len, 1)
            .equals(&output.datum_type, &input.datum_type)
            .equals(&padding.datum_type, DatumType::TDim)
            .equals(&input.rank, &output.rank)
            .equals(&padding.rank, 2)
            .equals(&padding.shape[0], input.rank.bex().to_dim())
            .equals(&padding.shape[1], 2.to_dim())
            .given(&input.rank, move |solver, rank: isize| {
                (0..rank as usize).for_each(|d| {
                    solver.equals(
                        &output.shape[d],
                        input.shape[d].bex()
                            + padding.value[d][0].bex().to_dim()
                            + padding.value[d][1].bex().to_dim(),
                    ); // FIXME
                });
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    use Tensor;

    #[test]
    fn pad_0() {
        let inputs = tvec![
            Tensor::from(arr2(&[[1, 2, 3], [4, 5, 6]])).into(),
            Tensor::from(arr2(&[[1, 1], [2, 2]])).into(),
        ];

        let expected: TVec<_> = tvec!(
            Tensor::from(arr2(&[
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 2, 3, 0, 0],
                [0, 0, 4, 5, 6, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ])).into()
        );

        assert_eq!(Pad::<i32>::new().eval(inputs).unwrap(), expected);
    }
}
