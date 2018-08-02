use ndarray::prelude::*;
// use num_traits::ToPrimitive;

mod concatv2;
mod fill;
mod pad;
mod pack;
mod squeeze;
mod strided_slice;

use ops::prelude::*;
use analyser::interface::*;
use tfpb::types::DataType;

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
    reg.insert("Squeeze", squeeze::squeeze);
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
    fn rules<'r, 'p: 'r, 's: 'r>(&'s self, solver: &mut Solver<'r>, inputs: &'p TensorsProxy, outputs: &'p TensorsProxy) {
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

            .given(&dims.value, move |solver, index:Tensor| {
                let index = index.take_i32s().unwrap(); // already enforced
                let index = index.as_slice().unwrap()[0] as usize; //already enforced

                for i in 0..index {
                    solver.equals(&output.shape[i], &data.shape[i]);
                }

                solver.equals(&output.shape[index], 1);

                solver.given(&data.rank, move |solver, rank| {
                    for i in index..rank {
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
    fn rules<'r, 'p: 'r, 's: 'r>(&'s self, solver: &mut Solver<'r>, inputs: &'p TensorsProxy, outputs: &'p TensorsProxy) {
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

    fn infer_and_propagate(
        &self,
        inputs: Vec<TensorFact>,
        outputs: Vec<TensorFact>,
    ) -> Result<(Vec<TensorFact>, Vec<TensorFact>)> {
        use ops::InferenceOp;
        self.infer(inputs, outputs)
    }
}

impl InferenceRulesOp for Placeholder {
    /// Registers the inference rules of the operator.
    fn rules<'r, 'p: 'r, 's: 'r>(&'s self, solver: &mut Solver<'r>, inputs: &'p TensorsProxy, outputs: &'p TensorsProxy) {
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
}

impl InferenceRulesOp for Reshape {
    fn rules<'r, 'p: 'r, 's: 'r>(&'s self, solver: &mut Solver<'r>, inputs: &'p TensorsProxy, outputs: &'p TensorsProxy) {
        solver
            .equals(&inputs.len, 2)
            .equals(&outputs.len, 1)
            .equals(&inputs[1].datatype, DataType::DT_FLOAT)
            .equals(&outputs[0].datatype, DataType::DT_INT32)
            .equals(&inputs[1].rank, 1)
            .equals(&inputs[0].datatype, &outputs[0].datatype)
            .given(&inputs[0].rank, move |solver, input_rank| {
                solver.given(&inputs[1].value, move |solver, dims:Vec<usize>| {
                    let dims:Vec<i32> = dims.into_iter().map(|d| d as i32).collect();
                    let shape = Reshape::true_dims(dims, input_rank);
                    solver.equals(&outputs[0].shape, ShapeFact::from(shape));
                });
            });
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
    fn rules<'r, 'p: 'r, 's: 'r>(&'s self, solver: &mut Solver<'r>, inputs: &'p TensorsProxy, outputs: &'p TensorsProxy) {
        solver
            .equals(&inputs.len, 1)
            .equals(&outputs.len, 1)

            .equals(&outputs[0].datatype, DataType::DT_INT32)
            .equals(&outputs[0].rank, 1)
            .equals(&outputs[0].shape[0], &inputs[0].rank)

            .given(&inputs[0].shape, move |solver, shape: ShapeFact| {
                if !shape.open && shape.dims.iter().all(|d| *d != DimFact::Any) {
                    let shape = shape.dims.iter().map(|d| if let DimFact::Only(d)= d { *d as i32 } else { 1 }).collect();
                    let array1:Array1<i32> = Array1::from_vec(shape);
                    let tensor:Tensor = Tensor::from(array1);
                    solver.equals(&outputs[0].value, valuefact!(tensor));
                }
            })
            .given(&outputs[0].value, move |solver, shape:Tensor| {
                let shape:Vec<usize> = shape.take_i32s().unwrap().iter().map(|i:&i32| *i as usize).collect();
                for (ix,d) in shape.iter().enumerate() {
                    // hackish: if dim is 1, it may be the streaming
                    // dimension, so we don't infer
                    if *d != 1 {
                        solver.equals(&inputs[0].shape[ix], *d as isize);
                    }
                }
            });
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
            value: valuefact!(Tensor::i32s(&[3], &[1, 2, 3]).unwrap())
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
