use num_traits::Zero;
use tract_core::ndarray::{Array, ArrayD, ArrayView2, ArrayViewD};

use crate::model::ParsingContext;
use crate::tfpb::tensorflow::NodeDef;

use tract_core::infer::*;
use tract_core::internal::*;

#[derive(Debug, Clone, Default, new)]
pub struct Pad<T: Copy + Datum + Zero> {
    _phantom: PhantomData<T>,
}

pub fn pad(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let dtype = pb.get_attr_datum_type("T")?;
    Ok(boxed_new!(Pad(dtype)()))
}

impl<T: Copy + Datum + Zero> Pad<T> {
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
    T: Copy + Datum + Zero,
{
    fn name(&self) -> Cow<str> {
        "tf.Pad".into()
    }

    not_a_typed_op!();
}

impl<T: Copy + Datum + Zero> StatelessOp for Pad<T> {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (input, paddings) = args_2!(inputs);
        let input = input.to_array_view::<T>()?;
        let paddings = paddings.to_array_view::<i32>()?.into_dimensionality()?;
        Ok(tvec![Self::compute(&input, paddings, None)?.into_arc_tensor()])
    }
}

impl<T: Copy + Datum + Zero> InferenceRulesOp for Pad<T> {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        let input = &inputs[0];
        let padding = &inputs[1];
        let output = &outputs[0];
        check_input_arity(&inputs, 2)?;
        check_output_arity(&outputs, 1)?;
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

    as_op!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_0() {
        let inputs = tvec![rctensor2(&[[1, 2, 3], [4, 5, 6]]), rctensor2(&[[1, 1], [2, 2]]),];

        let expected: TVec<_> = tvec!(rctensor2(&[
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 2, 3, 0, 0],
            [0, 0, 4, 5, 6, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]));

        assert_eq!(Pad::<i32>::new().eval(inputs).unwrap(), expected);
    }
}
