use ndarray::prelude::*;
use tract_core::ops::prelude::*;

#[derive(Debug, Clone, new)]
pub struct Reshape<T: Datum>(PhantomData<T>);

pub fn reshape(pb: &::tfpb::node_def::NodeDef) -> TractResult<Box<Op>> {
    let dtype = pb.get_attr_datum_type("T")?;
    Ok(boxed_new!(Reshape(dtype)()))
}

impl<T: Datum> Reshape<T> {
    /// Computes a vector of dimensions from the `dims` input.
    /// This is needed because `dims` might contain some -1 indices, in which
    /// case we need to infer the value for that index.
    fn true_dims(dims: ArrayViewD<i32>, input_length: usize) -> Vec<usize> {
        let prod: usize = dims
            .iter()
            .filter(|a| **a != -1)
            .map(|&a| a as usize)
            .product();
        dims.iter()
            .map(|&a| {
                if a == -1 {
                    input_length / prod
                } else {
                    a as usize
                }
            }).collect()
    }
}

impl<T: Datum> Op for Reshape<T> {
    fn name(&self) -> &str {
        "tf.Reshape"
    }
}

impl<T: Datum> StatelessOp for Reshape<T> {
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let (input, dims) = args_2!(inputs);

        let input = input.to_array::<T>()?;
        let dims = dims.to_array_view::<i32>()?;
        let dims = Self::true_dims(dims, input.len());
        let output = input.into_shape(&*dims)?.into_dyn();
        Ok(tvec![output.into()])
    }
}

impl<T: Datum> InferenceRulesOp for Reshape<T> {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p SharedTensorsProxy,
        outputs: &'p SharedTensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 2)?;
        s.equals(&outputs.len, 1)?;
        s.equals(&inputs[0].datum_type, T::datum_type())?;
        s.equals(&inputs[1].datum_type, DatumType::I32)?;
        s.equals(&outputs[0].datum_type, T::datum_type())?;
        s.equals(&inputs[1].rank, 1)?;
        s.given_2(
            &inputs[0].shape,
            &inputs[1].value,
            move |solver, shape, dims| {
                let dims = dims.to_array_view::<i32>().unwrap(); // checked
                if shape.iter().all(|d| !d.is_stream()) {
                    let len = shape
                        .iter()
                        .map(|d| d.as_const().unwrap() as usize)
                        .product();
                    let shape = Self::true_dims(dims, len);
                    solver.equals(&outputs[0].shape, ShapeFact::from(shape))?;
                }
                Ok(())
            },
        )
    }
}
