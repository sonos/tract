use analyser::interface::*;
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
    fn true_dims(mut dims: Vec<i32>, input_length: usize) -> Vec<usize> {
        if dims.contains(&-1) {
            let prod: i32 = dims.iter().map(|a| *a).filter(|a| *a != -1i32).product();
            for a in dims.iter_mut() {
                if *a == -1 {
                    *a = input_length as i32 / prod;
                }
            }
        }

        dims.into_iter().map(|a| a as usize).collect()
    }
}

impl<T: Datum> Op for Reshape<T> {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: Vec<Value>) -> Result<Vec<Value>> {
        let (input, dims) = args_2!(inputs);

        let input = T::tensor_into_array(input.into_tensor())?;
        let dims = <i32 as Datum>::tensor_into_array(dims.into_tensor())?;
        let dims = Self::true_dims(dims.iter().cloned().collect(), input.len());

        let output = input.into_shape(&*dims)?.into_dyn();
        Ok(vec![T::array_into_tensor(output).into()])
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
            .given(&inputs[0].rank, move |solver, input_rank| {
                solver.given(&inputs[1].value, move |solver, dims: Tensor| {
                    let dims = <i32 as Datum>::tensor_into_array(dims).unwrap(); // checked
                    let shape = Self::true_dims(dims.into_iter().cloned().collect(), input_rank as usize);
                    solver.equals(&outputs[0].shape, ShapeFact::from(shape));
                });
            });
    }
}
