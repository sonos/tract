use std::marker::PhantomData;

use ndarray::{Array, ArrayD, ArrayView2, ArrayViewD};
use num_traits::Zero;

use tract_core::ops::prelude::*;
use tract_core::TractResult;

#[derive(Debug, Clone, Default, new)]
pub struct Pad<T: Datum + Zero> {
    _phantom: PhantomData<T>,
}

pub fn pad(pb: &crate::tfpb::node_def::NodeDef) -> TractResult<Box<Op>> {
    let dtype = pb.get_attr_datum_type("T")?;
    Ok(boxed_new!(Pad(dtype)()))
}

impl<T: Datum + Zero> Pad<T> {
    fn compute(
        input: &ArrayViewD<T>,
        paddings: ArrayView2<i32>,
        stream_dim: Option<usize>,
    ) -> TractResult<ArrayD<T>> {
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
    T: Datum + Zero,
{
    fn name(&self) -> Cow<str> {
        "tf.Pad".into()
    }
}

impl<T: Datum + Zero> StatelessOp for Pad<T> {
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let (input, paddings) = args_2!(inputs);
        let input = input.to_array_view::<T>()?;
        let paddings = paddings.to_array_view::<i32>()?.into_dimensionality()?;
        Ok(tvec![Self::compute(&input, paddings, None)?.into()])
    }
}

impl<T: Datum + Zero> InferenceRulesOp for Pad<T> {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p SharedTensorsProxy,
        outputs: &'p SharedTensorsProxy,
    ) -> InferenceResult {
        let input = &inputs[0];
        let padding = &inputs[1];
        let output = &outputs[0];
        s.equals(&inputs.len, 2)?;
        s.equals(&outputs.len, 1)?;
        s.equals(&output.datum_type, &input.datum_type)?;
        s.equals(&padding.datum_type, DatumType::TDim)?;
        s.equals(&input.rank, &output.rank)?;
        s.equals(&padding.rank, 2)?;
        s.equals(&padding.shape[0], input.rank.bex().to_dim())?;
        s.equals(&padding.shape[1], 2.to_dim())?;
        s.given(&input.rank, move |s, rank| {
            for d in 0..rank as usize {
                s.equals(
                    &output.shape[d],
                    input.shape[d].bex()
                        + padding.value[d][0].bex().to_dim()
                        + padding.value[d][1].bex().to_dim(),
                )?
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    use tract_core::Tensor;

    #[test]
    fn pad_0() {
        let inputs = tvec![
            Tensor::from(arr2(&[[1, 2, 3], [4, 5, 6]])).into(),
            Tensor::from(arr2(&[[1, 1], [2, 2]])).into(),
        ];

        let expected: TVec<_> = tvec!(Tensor::from(arr2(&[
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 2, 3, 0, 0],
            [0, 0, 4, 5, 6, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]))
        .into());

        assert_eq!(Pad::<i32>::new().eval(inputs).unwrap(), expected);
    }
}
