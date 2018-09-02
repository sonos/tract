use ndarray::prelude::*;
use analyser::rules::prelude::*;
use ops::prelude::*;

#[derive(Debug, Clone, new)]
pub struct Reshape<T: Datum>(PhantomData<T>);

pub fn reshape(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
    let dtype = pb.get_attr_datum_type("T")?;
    Ok(boxed_new!(Reshape(dtype)()))
}

impl<T: Datum> Reshape<T> {
    /// Computes a vector of dimensions from the `dims` input.
    /// This is needed because `dims` might contain some -1 indices, in which
    /// case we need to infer the value for that index.
    fn true_dims(dims: ArrayViewD<i32>, input_length: usize) -> Vec<usize> {
        let prod: usize = dims.iter()
            .filter(|a| **a != -1)
            .map(|&a| a as usize)
            .product();
        dims.iter().map(|&a| if a == -1 { input_length/prod } else { a as usize } ).collect()
    }
}

impl<T: Datum> Op for Reshape<T> {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Value>) -> Result<TVec<Value>> {
        let (input, dims) = args_2!(inputs);

        let input = input.into_array::<T>()?;
        let dims = dims.to_array_view::<i32>()?;
        let dims = Self::true_dims(dims, input.len());
        let output = input.into_shape(&*dims)?.into_dyn();
        Ok(tvec![output.into()])
    }

    /// Returns the attributes of the operation and their values.
    fn get_attributes(&self) -> HashMap<&'static str, Attr> {
        hashmap!{}
    }
}

impl<T: Datum> InferenceRulesOp for Reshape<T> {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) {
        solver
            .equals(&inputs.len, 2)
            .equals(&outputs.len, 1)
            .equals(&inputs[0].datum_type, T::datum_type())
            .equals(&inputs[1].datum_type, DatumType::I32)
            .equals(&outputs[0].datum_type, T::datum_type())
            .equals(&inputs[1].rank, 1)
            .given(&inputs[0].shape, move |solver, shape| {
                solver.given(&inputs[1].value, move |solver, dims: Tensor| {
                    let dims = dims.to_array_view::<i32>().unwrap(); // checked
                    if shape.iter().all(|d| !d.is_stream()) {
                        let len = shape.iter().map(|d| d.as_const().unwrap() as usize).product();
                        let shape = Self::true_dims(dims, len);
                        solver.equals(&outputs[0].shape, ShapeFact::from(shape));
                    }
                });
            });
    }
}
