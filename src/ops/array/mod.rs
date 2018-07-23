use std::marker::PhantomData;
use std::collections::HashMap;
use ndarray::prelude::*;
use num_traits::ToPrimitive;

mod concatv2;
mod fill;
mod pad;
mod pack;
mod strided_slice;

use ops::prelude::*;
use analyser::interface::*;
use tfpb::types::DataType;
use tensor::Datum;
use {Result, Tensor};

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("ConcatV2", concatv2::build);
    reg.insert("ExpandDims", ExpandDims::build);
    reg.insert("Identity", Identity::build);
    reg.insert("Fill", fill::fill);
    reg.insert("Pack", pack::pack);
    reg.insert("Pad", pad::pad);
    reg.insert("Placeholder", Placeholder::build);
    reg.insert("Reshape", Reshape::build);
    reg.insert("Shape", Shape::build);
    reg.insert("Squeeze", squeeze);
    reg.insert("StridedSlice", strided_slice::build);
}

#[derive(Debug, Clone)]
pub struct ExpandDims;

impl ExpandDims {
    pub fn build(_pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        Ok(Box::new(ExpandDims))
    }
}

impl Op for ExpandDims {
    /// Returns the attributes of the operation and their values.
    fn get_attributes(&self) -> HashMap<&'static str, Attr> {
        hashmap!{}
    }

    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: Vec<TensorView>) -> Result<Vec<TensorView>> {
        let (data, dims) = args_2!(inputs);
        let data = data.into_tensor()
            .take_f32s()
            .ok_or("Expected a f32 matrix")?;
        let dims = dims.as_i32s().ok_or("Expected a i32 matrix")?;
        let mut shape = data.shape().to_vec();
        for d in dims.iter() {
            if *d >= 0 {
                shape.insert(*d as usize, 1);
            } else {
                Err(format!("unimplemented ExpandDims with negative parameter"))?
            }
        }
        Ok(vec![Tensor::from(data.into_shape(shape)?).into()])
    }

    /// Evaluates one step of the operation on the given input tensors.
    fn step(
        &self,
        mut inputs: Vec<(Option<usize>, Option<TensorView>)>,
        _: &mut Box<OpBuffer>,
    ) -> Result<Option<Vec<TensorView>>> {
        let (data, dims) = args_2!(inputs);

        if dims.0.is_some() || dims.1.is_none() {
            bail!("Dims input should not be streamed.");
        }

        let dims = dims.1.unwrap();

        match data.1 {
            None => Ok(None),
            Some(tv) => Ok(Some(self.eval(vec![tv, dims])?))
        }
    }
}

impl InferenceRulesOp for ExpandDims {
    fn rules<'r, 'p: 'r>(&self, solver: &mut Solver<'r>, inputs: &'p TensorsProxy, outputs: &'p TensorsProxy) {
        let data = &inputs[0];
        let dims = &inputs[1];
        let output = &outputs[0];

        solver
            .equals(&inputs.len, 2)
            .equals(&outputs.len, 1)

            .equals(&dims.datatype, DataType::DT_INT32)
            .equals(&dims.rank, 0)

            .equals(&data.datatype, &output.datatype)
            .equals_zero(wrap![&data.rank, 1, (-1, &output.rank)])

            .given(&dims.value, move |solver, index| {
                let index = index.to_usize().unwrap();

                for i in 0..index {
                    solver.equals(&output.shape[i], &data.shape[i]);
                }

                solver.equals(&output.shape[index], 1);

                solver.given(&data.rank, move |solver, rank| {
                    let rank = rank.to_usize().unwrap();

                    for i in 0..rank {
                        solver.equals(&output.shape[i + 1], &data.shape[i]);
                    }
                });
            });
    }
}

#[derive(Debug, Clone)]
pub struct Identity;

impl Identity {
    pub fn build(_: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        Ok(Box::new(Identity))
    }
}

impl Op for Identity {
    /// Returns the attributes of the operation and their values.
    fn get_attributes(&self) -> HashMap<&'static str, Attr> {
        hashmap!{}
    }

    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: Vec<TensorView>) -> Result<Vec<TensorView>> {
        Ok(inputs)
    }

    /// Evaluates one step of the operation on the given input tensors.
    fn step(
        &self,
        mut inputs: Vec<(Option<usize>, Option<TensorView>)>,
        _: &mut Box<OpBuffer>,
    ) -> Result<Option<Vec<TensorView>>> {
        let input = args_1!(inputs);
        match input.1 {
            None => Ok(None),
            Some(tv) => Ok(Some(self.eval(vec![tv])?))
        }
    }
}

impl InferenceRulesOp for Identity {
    fn rules<'r, 'p: 'r>(&self, solver: &mut Solver<'r>, inputs: &'p TensorsProxy, outputs: &'p TensorsProxy) {
        solver
            .equals(&inputs.len, 1)
            .equals(&outputs.len, 1)
            .equals(&inputs[0].datatype, &outputs[0].datatype)
            .equals(&inputs[0].shape, &outputs[0].shape);
    }
}

#[derive(Debug, Clone)]
pub struct Placeholder {
    dtype: DataType,
}

impl Placeholder {
    pub fn build(node: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        Ok(Box::new(Placeholder {
            dtype: node.get_attr_datatype("dtype")?,
        }))
    }
}

impl Op for Placeholder {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, _inputs: Vec<TensorView>) -> Result<Vec<TensorView>> {
        panic!("Placeholder should not get evaluated")
    }

    /// Returns the attributes of the operation and their values.
    fn get_attributes(&self) -> HashMap<&'static str, Attr> {
        hashmap!{
            "dtype" => Attr::DataType(self.dtype)
        }
    }
}

impl InferenceRulesOp for Placeholder {
    /// Registers the inference rules of the operator.
    fn rules<'r, 'p: 'r>(&self, solver: &mut Solver<'r>, inputs: &'p TensorsProxy, outputs: &'p TensorsProxy) {
        solver
            .equals(&inputs.len, 0)
            .equals(&outputs.len, 1)
            .equals(&outputs[0].datatype, self.dtype);
    }
}

#[derive(Debug, Clone)]
pub struct Reshape {}

impl Reshape {
    pub fn build(_pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        Ok(Box::new(Reshape {}))
    }

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

impl Op for Reshape {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: Vec<TensorView>) -> Result<Vec<TensorView>> {
        let (input, dims) = args_2!(inputs);

        let input = input
            .into_tensor()
            .take_f32s()
            .ok_or("Expected a f32 matrix")?;

        let dims = Reshape::true_dims(
            dims.as_i32s()
                .ok_or("Expected a i32 matrix")?
                .iter()
                .cloned()
                .collect(),
            input.len(),
        );

        Ok(vec![
            Tensor::from(input.into_shape(&*dims)?.into_dyn()).into(),
        ])
    }

    /// Returns the attributes of the operation and their values.
    fn get_attributes(&self) -> HashMap<&'static str, Attr> {
        hashmap!{}
    }

    /*
    /// Infers properties about the input and output tensors.
    /// TODO(liautaud): This is ugly, rewrite using the solver.
    fn infer(
        &self,
        inputs: Vec<TensorFact>,
        mut outputs: Vec<TensorFact>,
    ) -> Result<(Vec<TensorFact>, Vec<TensorFact>)> {
        if inputs.len() != 2 {
            bail!("Reshape operation only supports two inputs.");
        }

        if outputs.len() < 1 {
            bail!("Reshape operation only supports one output.");
        }

        let input = unify(&inputs[0], &TensorFact {
            datatype: outputs[0].datatype,
            ..TensorFact::new()
        })?;

        let shape = unify(&inputs[1], &TensorFact {
            datatype: typefact!(DataType::DT_INT32),
            ..TensorFact::new()
        })?;

        let output: Result<_> = {
            let value = &inputs[1].value.concretize();

            if let Some(value) = value {
                // If we don't know the actual value, we can still compute the shape.
                let dims: Vec<_> = value
                    .as_i32s()
                    .ok_or("Expected a i32 matrix")?
                    .iter()
                    .cloned()
                    .collect();

                match &inputs[0].shape.concretize() {
                    // If we know the concrete shape of the input, we get the output shape.
                    Some(shape) => Ok(Some(TensorFact {
                        datatype: inputs[0].datatype,
                        shape: Reshape::true_dims(dims, shape[0]).iter().cloned().collect(),
                        value: valuefact!(_),
                    })),

                    // If we don't know anything about the output, but know the value of
                    // dims and it doesn't contain -1 (e.g. we don't have to guess some
                    // of the output dimensions), we can also compute the output shape.
                    _ if !dims.contains(&-1) => Ok(Some(TensorFact {
                        datatype: inputs[0].datatype,
                        shape: dims.into_iter().map(|d| d as usize).collect(),
                        value: valuefact!(_),
                    })),

                    _ => Ok(None)
                }
            } else {
                Ok(None)
            }
        };

        let output = match output? {
            Some(fact) => unify(&outputs[0], &fact)?,
            None => outputs.remove(0),
        };

        Ok((vec![input, shape], vec![output]))
    }
    */
}

impl InferenceRulesOp for Reshape {
    fn rules<'r, 'p: 'r>(&self, solver: &mut Solver<'r>, inputs: &'p TensorsProxy, outputs: &'p TensorsProxy) {
        unimplemented!()
    }
}

#[derive(Debug, Clone)]
pub struct Shape;

impl Shape {
    pub fn build(_pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        Ok(Box::new(Shape))
    }
}

impl Op for Shape {
    /// Returns the attributes of the operation and their values.
    fn get_attributes(&self) -> HashMap<&'static str, Attr> {
        hashmap!{}
    }

    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: Vec<TensorView>) -> Result<Vec<TensorView>> {
        let data = inputs[0].as_f32s().ok_or("Expect input #0 to be f32")?;
        let shape: Vec<i32> = data.shape().into_iter().map(|s| *s as i32).collect();
        Ok(vec![Tensor::from(Array1::from_vec(shape)).into()])
    }
}

impl InferenceRulesOp for Shape {
    fn rules<'r, 'p: 'r>(&self, solver: &mut Solver<'r>, inputs: &'p TensorsProxy, outputs: &'p TensorsProxy) {
        solver
            .equals(&inputs.len, 1)
            .equals(&outputs.len, 1)

            .equals(&outputs[0].datatype, DataType::DT_INT32)
            .equals(&outputs[0].rank, 1)
            .equals(&outputs[0].shape[0], &inputs[0].rank)

            .given(&inputs[0].rank, move |solver, ir| {
                for i in 0..ir.to_usize().unwrap() {
                    solver.equals(&outputs[0].value[i], &inputs[0].shape[i]);
                }
            });
    }
}

pub fn squeeze(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
    let mut dims = pb.get_attr_opt_list_int("squeeze_dims")?;
    if let Some(ref mut dims) = dims {
        dims.sort();
        dims.reverse();
    }
    let t = pb.get_attr_datatype("T")?;
    Ok(boxed_new!(Squeeze(t)(dims)))
}

#[derive(Debug, Clone, new)]
pub struct Squeeze<T: Datum> {
    dims: Option<Vec<isize>>,
    t: PhantomData<T>,
}

impl<T: Datum> Squeeze<T> {
    /// Removes the dimensions of size 1 from the given shape vector.
    fn squeeze_shape(&self, mut shape: Vec<usize>, stream_dim: Option<usize>) -> Result<Vec<usize>> {
        if let Some(ref dims) = self.dims {
            for d in dims {
                if *d >= 0 {
                    shape.remove(*d as usize);
                } else {
                    Err(format!("unimplemented Squeeze with negative parameter"))?
                }
            }
            Ok(shape)
        } else {
            Ok(shape.into_iter().enumerate().filter(|&(ix, d)| stream_dim == Some(ix) || d != 1).map(|(_,d)| d).collect())
        }
    }
}

impl<T: Datum> Op for Squeeze<T> {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: Vec<TensorView>) -> Result<Vec<TensorView>> {
        let data = inputs[0].as_f32s().ok_or("Expect input #0 to be f32")?;
        let shape = self.squeeze_shape(data.shape().to_vec(), None)?;
        Ok(vec![Tensor::from(data.clone().into_shape(shape)?).into()])
    }
/*
    /// Infers properties about the input and output tensors.
    /// TODO(liautaud): This is ugly, rewrite using the solver.
    fn infer(
        &self,
        inputs: Vec<TensorFact>,
        mut outputs: Vec<TensorFact>,
    ) -> Result<(Vec<TensorFact>, Vec<TensorFact>)> {
        if inputs.len() != 1 {
            bail!("Squeeze operation only supports one input.");
        }

        if outputs.len() != 1 {
            bail!("Squeeze operation only supports one output.");
        }

        let output: Result<_> = {
            // We can't say anything interesting if there are unknown dimensions,
            // because they could turn out to be Only(1), and so Squeeze would
            // have to remove them.
            let shape = match inputs[0].shape.concretize() {
                Some(shape) => self.squeeze_shape(shape, None)?.iter().cloned().collect(),
                None => shapefact![..],
            };

            let output = TensorFact {
                datatype: inputs[0].datatype,
                shape,
                value: valuefact!(_),
            };

            Ok(Some(vec![output]))
        };

        let output = match output? {
            Some(v) => unify(&outputs[0], &v[0])?,
            None => outputs.remove(0),
        };

        let input = unify(&inputs[0], &TensorFact {
            datatype: output.datatype,
            shape: shapefact![..],
            value: valuefact!(_),
        })?;

        Ok((vec![input], vec![output]))
    }
*/

    /// Returns the attributes of the operation and their values.
    fn get_attributes(&self) -> HashMap<&'static str, Attr> {
        if let Some(dim) = self.dims.as_ref() {
            hashmap! { "dims" => Attr::IsizeVec(dim.clone()) }
        } else {
            hashmap!{ }
        }
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
            let shape = self.squeeze_shape(chunk.shape().to_vec(), Some(stream))?;
            Ok(Some(vec!( T::array_into_tensor(chunk.into_shape(shape)?).into() )))
        } else {
            Ok(None)
        }
    }
}

impl<T:Datum> InferenceRulesOp for Squeeze<T> {
    fn rules<'r, 'p: 'r>(&self, solver: &mut Solver<'r>, inputs: &'p TensorsProxy, outputs: &'p TensorsProxy) {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tfpb::node;

    #[test]
    fn shape_inference_1() {
        let input = TensorFact {
            datatype: typefact!(DataType::DT_FLOAT),
            shape: shapefact![1, _, _; ..],
            value: valuefact!(_)
        };

        let output = TensorFact {
            datatype: typefact!(DataType::DT_INT32),
            shape: shapefact![_],
            value: valuefact!(_)
        };

        assert_forward!(Shape::build(&node()).unwrap(), input, output);
    }

    #[test]
    fn shape_inference_2() {
        let input = TensorFact {
            datatype: typefact!(DataType::DT_FLOAT),
            shape: shapefact![1, _, _],
            value: valuefact!(_)
        };

        let output = TensorFact {
            datatype: typefact!(DataType::DT_INT32),
            shape: shapefact![3],
            value: valuefact!(_)
        };

        assert_forward!(Shape::build(&node()).unwrap(), input, output);
    }

    #[test]
    fn shape_inference_3() {
        let input = TensorFact {
            datatype: typefact!(DataType::DT_FLOAT),
            shape: shapefact![1, 2, 3],
            value: valuefact!(_)
        };

        let output = TensorFact {
            datatype: typefact!(DataType::DT_INT32),
            shape: shapefact![3],
            // FIXME(liautaud)
            value: valuefact!(_)
        };

        assert_forward!(Shape::build(&node()).unwrap(), input, output);
    }

    #[test]
    fn shape_inference_4() {
        let input = TensorFact {
            datatype: typefact!(_),
            shape: shapefact![1, 2, 3],
            value: valuefact!(_)
        };

        let output = TensorFact {
            datatype: typefact!(DataType::DT_INT32),
            shape: shapefact![3],
            value: valuefact!(Tensor::i32s(&[3], &[1, 2, 3]).unwrap())
        };

        assert_backward!(Shape::build(&node()).unwrap(), input, output);
    }
}
