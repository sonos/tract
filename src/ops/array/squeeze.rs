use ops::prelude::*;
use analyser::interface::*;

pub fn squeeze(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
    let mut squeeze_dims = pb.get_attr_opt_list_int("squeeze_dims")?;
    if let Some(ref mut squeeze_dims) = squeeze_dims {
        squeeze_dims.sort();
        squeeze_dims.reverse();
    }
    let t = pb.get_attr_datatype("T")?;
    Ok(boxed_new!(Squeeze(t)(squeeze_dims)))
}

#[derive(Debug, Clone, new)]
pub struct Squeeze<T: Datum> {
    squeeze_dims: Option<Vec<isize>>,
    t: PhantomData<T>,
}

impl<T: Datum> Squeeze<T> {

    fn squeezable(&self, ix: usize, d:usize, stream_dim: Option<usize>) -> bool {
        stream_dim != Some(ix) && d == 1
            && self.squeeze_dims.as_ref().map(|squeeze_dims| squeeze_dims.contains(&(ix as _))).unwrap_or(true)
    }

    /// Removes the dimensions of size 1 from the given shape vector.
    fn squeeze_shape(&self, shape: &[usize], stream_dim: Option<usize>) -> Vec<usize> {
        shape.iter().enumerate().filter_map(|(ix, d)| if self.squeezable(ix, *d, stream_dim) { None } else { Some(*d) }).collect()
    }
}

impl<T: Datum> Op for Squeeze<T> {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: Vec<TensorView>) -> Result<Vec<TensorView>> {
        let input = args_1!(inputs);
        let data = T::tensor_into_array(input.into_tensor())?;
        let shape = self.squeeze_shape(data.shape(), None);
        Ok(vec![T::array_into_tensor(data.clone().into_shape(shape)?).into()])
    }

    /// Returns the attributes of the operation and their values.
    fn get_attributes(&self) -> HashMap<&'static str, Attr> {
        let mut attrs = hashmap!{ "T" => Attr::DataType(T::datatype()) };
        if let Some(dim) = self.squeeze_dims.as_ref() {
            attrs.insert("squeeze_dims", Attr::IsizeVec(dim.clone()));
        }
        attrs
    }

    /// Evaluates one step of the operation on the given input tensors.
    fn step(
        &self,
        mut inputs: Vec<(Option<usize>, Option<TensorView>)>,
        _buffer: &mut Box<OpBuffer>,
    ) -> Result<Option<Vec<TensorView>>> {
        let input = args_1!(inputs);
        if let (Some(stream), Some(chunk)) = input {
            let chunk = T::tensor_into_array(chunk.into_tensor())?;
            let shape = self.squeeze_shape(chunk.shape(), Some(stream));
            Ok(Some(vec!( T::array_into_tensor(chunk.into_shape(shape)?).into() )))
        } else {
            Ok(None)
        }
    }
}

impl<T:Datum> InferenceRulesOp for Squeeze<T> {
    fn rules<'r, 'p: 'r, 's: 'r>(&'s self, solver: &mut Solver<'r>, inputs: &'p TensorsProxy, outputs: &'p TensorsProxy) {
        solver
            .equals(&inputs.len, 1)
            .equals(&outputs.len, 1)
            .equals(&inputs[0].datatype, &outputs[0].datatype)
            .equals(&inputs[0].datatype, T::datatype())
            .given(&inputs[0].shape, move |solver, shape:ShapeFact| {
                if !shape.dims.iter().any(|d| *d == DimFact::Any) {
                    let stream_dim = shape.dims.iter().position(|d| *d == DimFact::Streamed);
                    let shape:Vec<DimFact> = shape.dims.into_iter().enumerate().filter_map(|(ix, d)|
                        if self.squeezable(ix, d.concretize().unwrap_or(1), stream_dim) { None } else { Some(d) } 
                    ).collect();
                    let fact = ShapeFact::closed(shape);
                    solver.equals(&outputs[0].shape, fact);
                }
            });
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use super::*;
    use ndarray::*;
    use Tensor;

    fn run<I>(op: Squeeze<i32>, input: I) -> Tensor
    where
        I: Into<Tensor>,
    {
        op.eval(vec![
            input.into().into(),
        ]).unwrap()
            .pop()
            .unwrap()
            .into_tensor()
    }

    #[test]
    fn squeeze_1() {
        assert_eq!(
            run(
                Squeeze::new(None),
                Array::from_elem([1, 2, 1, 3, 1, 1], 0)
            ).shape(),
            &[2, 3]
        );
    }

    #[test]
    fn squeeze_2() {
        assert_eq!(
            run(
                Squeeze::new(Some(vec!(2, 4))),
                Array::from_elem([1, 2, 1, 3, 1, 1], 0)
            ).shape(),
            &[1, 2, 3, 1]
        );
    }
}
