use std::collections::HashMap;
use std::collections::VecDeque;
use ndarray::prelude::*;
use std::iter::repeat;

mod pack;
mod strided_slice;

use ops::{Attr, Op, OpRegister, TensorView};
use analyser::helpers::infer_forward_concrete;
use analyser::helpers::most_specific_shape;
use analyser::{ShapeFact, TensorFact, ValueFact};
use tfpb::types::DataType;
use {Result, Tensor};

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("ConcatV2", ConcatV2::build);
    reg.insert("ExpandDims", ExpandDims::build);
    reg.insert("Identity", Identity::build);
    reg.insert("Pack", pack::pack);
    reg.insert("Placeholder", Placeholder::build);
    reg.insert("Reshape", Reshape::build);
    reg.insert("Shape", Shape::build);
    reg.insert("Squeeze", Squeeze::build);
    reg.insert("StridedSlice", strided_slice::build);
}

#[derive(Debug)]
pub struct ConcatV2 {
    n: usize,
}

impl ConcatV2 {
    pub fn build(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        Ok(Box::new(ConcatV2 {
            n: pb.get_attr_int("N")?,
        }))
    }
}

impl Op for ConcatV2 {
    /// Returns the attributes of the operation and their values.
    fn get_attributes(&self) -> HashMap<&'static str, Attr> {
        hashmap!{
            "n" => Attr::Usize(self.n),
        }
    }

    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: Vec<TensorView>) -> Result<Vec<TensorView>> {
        let axis: i32 = inputs[self.n]
            .as_i32s()
            .ok_or("Expected a i32 matrix")?
            .iter()
            .next()
            .unwrap()
            .clone();
        let mats: Vec<_> = inputs[0..self.n]
            .iter()
            .map(|mat| mat.as_f32s().unwrap().view())
            .collect();
        let result = ::ndarray::stack(Axis(axis as usize), &*mats)?;
        let result = Tensor::from(result);

        Ok(vec![result.into()])
    }

    /// Evaluates one step of the operation on the given input tensors.
    fn step(
        &self,
        mut inputs: Vec<(Option<usize>, Option<TensorView>)>,
        buffer: &mut Vec<VecDeque<TensorView>>,
    ) -> Result<Option<Vec<TensorView>>> {
        // According to https://www.tensorflow.org/api_docs/python/tf/concat,
        // the number of dimensions of each input tensor must match, and all
        // dimensions except `axis` must be equal. In particular, this means
        // that all the input tensors must have the same streaming dimension.
        // That leaves us with two cases:
        // - Either all the tensors are streamed along `axis`, in which case
        //   we push every slice we receive as input directly to the output.
        // - Or they are streamed along another dimension, so we buffer them
        //   until we have a chunk for each, and we push their concatenation
        //   as the output chunk.

        if inputs[self.n].0.is_some() || inputs[self.n].1.is_none() {
            bail!("Axis input should not be streamed.");
        }

        let axis_tensor = inputs[self.n].1.take().unwrap();
        let axis: i32 = axis_tensor
            .as_i32s()
            .ok_or("Expected a i32 matrix")?
            .iter()
            .next()
            .unwrap()
            .clone();

        if inputs[0..self.n].iter().all(|i| i.0 == Some(axis as usize)) {
            // All the input tensors are streamed along `axis`.
            let chunk = inputs[0..self.n].iter_mut()
                .find(|i| i.1.is_some())
                .unwrap()
                .1.take()
                .unwrap();

            Ok(Some(vec![chunk]))
        } else {
            // All the input tensors are streamed along a non-`axis` dimension.
            initialize_buffer!(buffer, self.n);
            append_buffer!(buffer, inputs[0..self.n]);

            if buffer.iter().any(|b| b.is_empty()) {
                Ok(None)
            } else {
                let mut chunks = buffer
                    .iter_mut()
                    .map(|b| b.pop_front().unwrap())
                    .collect::<Vec<_>>();

                chunks.push(axis_tensor);

                Ok(Some(self.eval(chunks)?))
            }
        }
    }

    /// Infers properties about the output tensors from the input tensors.
    fn infer_forward(&self, inputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>> {
        if inputs.len() < 2 {
            bail!("Concat operation needs at least two inputs.");
        }

        if let Some(output) = infer_forward_concrete(self, &inputs)? {
            return Ok(Some(output));
        }

        // If we don't know the actual value, we can still compute the shape.
        let axis: i32 = unwrap_or_none!(inputs[self.n].value.concretize())
            .as_i32s()
            .ok_or("Expected a i32 matrix")?
            .iter()
            .next()
            .unwrap()
            .clone();

        let shapes = inputs[0..self.n].iter().map(|t| &t.shape);

        // We get the most specific shape, and replace the axis with an unknown.
        // TODO(liautaud): Improve this to check whether the shapes actually match,
        //                 and sum the dimension over all the vectors instead of
        //                 just returning an unknown when possible.
        let shape = match most_specific_shape(shapes)? {
            Some(s) => {
                let mut dims = s.dims.clone();
                dims[axis as usize] = dimfact!(_);
                ShapeFact::closed(dims)
            }

            None => shapefact![..],
        };

        let output = TensorFact {
            datatype: inputs[0].datatype,
            shape,
            value: valuefact!(_),
        };

        Ok(Some(vec![output]))
    }

    /// Infers properties about the input tensors from the output tensors.
    fn infer_backward(&self, outputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>> {
        if outputs.len() < 1 {
            bail!("Concat operation only supports one output.");
        }

        // TODO(liautaud): Implement something here.
        Ok(None)
    }
}

#[derive(Debug)]
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
        _: &mut Vec<VecDeque<TensorView>>,
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

    /// Infers properties about the output tensors from the input tensors.
    fn infer_forward(&self, inputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>> {
        if inputs.len() != 2 {
            bail!("ExpandDims operation only supports two inputs.");
        }

        if let Some(output) = infer_forward_concrete(self, &inputs)? {
            return Ok(Some(output));
        }

        // If we don't know the actual value, we can still compute the shape.
        let input_shape = &inputs[0].shape;
        let mut dims: Vec<_> = unwrap_or_none!(inputs[1].value.concretize())
            .as_i32s()
            .ok_or("Expected a i32 matrix")?
            .iter()
            .map(|i| *i as usize)
            .collect();

        dims.sort();

        let mut output_dims = input_shape.dims.clone();
        for index in dims {
            if index > output_dims.len() && !input_shape.open {
                bail!("Can't insert a new dimension when index > input_dim.");
            } else if index > output_dims.len() {
                let current_dim = output_dims.len();
                output_dims.extend(repeat(dimfact!(_)).take(index - current_dim));
                output_dims.push(dimfact!(1));
            } else {
                output_dims.insert(index, dimfact!(1));
            }
        }

        let output = TensorFact {
            datatype: inputs[0].datatype,
            shape: if input_shape.open {
                ShapeFact::open(output_dims)
            } else {
                ShapeFact::closed(output_dims)
            },
            value: valuefact!(_),
        };

        Ok(Some(vec![output]))
    }

    /// Infers properties about the input tensors from the output tensors.
    fn infer_backward(&self, outputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>> {
        if outputs.len() < 1 {
            bail!("ExpandDims operation only supports one output.");
        }

        let data = TensorFact {
            datatype: outputs[0].datatype,
            shape: shapefact![..],
            value: valuefact!(_),
        };

        let dims = TensorFact {
            datatype: typefact!(DataType::DT_INT32),
            shape: shapefact![..],
            value: valuefact!(_),
        };

        Ok(Some(vec![data, dims]))
    }
}

#[derive(Debug)]
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
        _: &mut Vec<VecDeque<TensorView>>,
    ) -> Result<Option<Vec<TensorView>>> {
        let input = args_1!(inputs);
        match input.1 {
            None => Ok(None),
            Some(tv) => Ok(Some(self.eval(vec![tv])?))
        }
    }

    /// Infers properties about the output tensors from the input tensors.
    fn infer_forward(&self, inputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>> {
        if inputs.len() != 1 {
            bail!("Identity operation only supports one input.");
        }

        Ok(Some(inputs.into_iter().cloned().collect()))
    }

    /// Infers properties about the input tensors from the output tensors.
    fn infer_backward(&self, outputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>> {
        if outputs.len() < 1 {
            bail!("Identity operation only supports one output.");
        }

        Ok(Some(outputs.into_iter().cloned().collect()))
    }
}

#[derive(Debug)]
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

    /// Infers properties about the output tensors from the input tensors.
    fn infer_forward(&self, _inputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>> {
        let output = TensorFact {
            datatype: typefact!(self.dtype),
            shape: shapefact![..],
            value: valuefact!(_),
        };

        Ok(Some(vec![output]))
    }

    /// Infers properties about the input tensors from the output tensors.
    fn infer_backward(&self, _outputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>> {
        debug!("Placeholder operation is a leaf, nothing to infer backwards.");
        Ok(None)
    }
}

#[derive(Debug)]
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

    /// Infers properties about the output tensors from the input tensors.
    fn infer_forward(&self, inputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>> {
        if inputs.len() != 2 {
            bail!("Reshape operation only supports two inputs.");
        }

        if let Some(output) = infer_forward_concrete(self, &inputs)? {
            return Ok(Some(output));
        }

        // If we don't know the actual value, we can still compute the shape.
        let dims: Vec<_> = unwrap_or_none!(inputs[1].value.concretize())
            .as_i32s()
            .ok_or("Expected a i32 matrix")?
            .iter()
            .cloned()
            .collect();

        let output = match &inputs[0].shape.concretize() {
            // If we know the concrete shape of the input, we get the output shape.
            Some(shape) => TensorFact {
                datatype: inputs[0].datatype,
                shape: Reshape::true_dims(dims, shape[0]).iter().collect(),
                value: valuefact!(_),
            },

            // If we don't know anything about the output, but know the value of
            // dims and it doesn't contain -1 (e.g. we don't have to guess some
            // of the output dimensions), we can also compute the output shape.
            _ if !dims.contains(&-1) => TensorFact {
                datatype: inputs[0].datatype,
                shape: dims.into_iter().map(|d| d as usize).collect(),
                value: valuefact!(_),
            },

            _ => {
                return Ok(None);
            }
        };

        Ok(Some(vec![output]))
    }

    /// Infers properties about the input tensors from the output tensors.
    fn infer_backward(&self, outputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>> {
        if outputs.len() < 1 {
            bail!("Reshape operation only supports one output.");
        }

        let input = TensorFact {
            datatype: outputs[0].datatype,
            shape: shapefact![..],
            value: valuefact!(_),
        };

        let shape = TensorFact {
            datatype: typefact!(DataType::DT_INT32),
            shape: shapefact![_],
            value: valuefact!(_),
        };

        Ok(Some(vec![input, shape]))
    }
}

#[derive(Debug)]
pub struct Shape;

impl Shape {
    pub fn build(_pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        Ok(Box::new(Shape))
    }
}

impl Op for Shape {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: Vec<TensorView>) -> Result<Vec<TensorView>> {
        let data = inputs[0].as_f32s().ok_or("Expect input #0 to be f32")?;
        let shape: Vec<i32> = data.shape().into_iter().map(|s| *s as i32).collect();
        Ok(vec![Tensor::from(Array1::from_vec(shape)).into()])
    }

    /// Infers properties about the output tensors from the input tensors.
    fn infer_forward(&self, inputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>> {
        if inputs.len() != 1 {
            bail!("Shape operation only supports one input.");
        }

        // We don't care about the concrete value, just the shape.
        let shape: Vec<_> = unwrap_or_none!(inputs[0].shape.concretize())
            .into_iter()
            .map(|d| d as i32)
            .collect();
        let rank = shape.len();
        let value = Tensor::from(Array1::from_vec(shape)).into();

        // The output is the shape of the input.
        // The shape of the output is the rank of the input.
        let output = TensorFact {
            datatype: typefact!(DataType::DT_INT32),
            shape: shapefact![rank],
            value: valuefact!(value),
        };

        Ok(Some(vec![output]))
    }

    /// Returns the attributes of the operation and their values.
    fn get_attributes(&self) -> HashMap<&'static str, Attr> {
        hashmap!{}
    }

    /// Infers properties about the input tensors from the output tensors.
    fn infer_backward(&self, outputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>> {
        if outputs.len() < 1 {
            bail!("Shape operation only supports one output.");
        }

        let dimensions: ShapeFact = match &outputs[0].value {
            // If we know the output value, we can infer the shape of the input.
            ValueFact::Only(v) => v.clone()
                .take_i32s()
                .ok_or("Shape operation should produce a 1-D integer tensor.")?
                .into_dimensionality::<Ix1>()?
                .into_iter()
                .map(|d| *d as usize)
                .collect(),

            // Otherwise, we can only infer the rank of the input.
            ValueFact::Any => {
                let shape = unwrap_or_none!(outputs[0].shape.concretize());

                if shape.len() != 1 {
                    bail!("Shape operation should produce a 1-D integer tensor.");
                }

                ShapeFact::closed(vec![dimfact!(_); shape[0]])
            }
        };

        Ok(Some(vec![TensorFact {
            datatype: typefact!(_),
            shape: dimensions,
            value: valuefact!(_),
        }]))
    }
}

#[derive(Debug)]
pub struct Squeeze {
    dims: Vec<isize>,
}

impl Squeeze {
    pub fn build(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        let mut dims = pb.get_attr_list_int("squeeze_dims")?;
        dims.sort();
        dims.reverse();
        Ok(Box::new(Squeeze { dims }))
    }

    /// Removes the dimensions of size 1 from the given shape vector.
    fn squeeze_shape(&self, mut shape: Vec<usize>) -> Result<Vec<usize>> {
        for d in &self.dims {
            if *d >= 0 {
                shape.remove(*d as usize);
            } else {
                Err(format!("unimplemented Squeeze with negative parameter"))?
            }
        }

        Ok(shape)
    }
}

impl Op for Squeeze {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: Vec<TensorView>) -> Result<Vec<TensorView>> {
        let data = inputs[0].as_f32s().ok_or("Expect input #0 to be f32")?;
        let shape = self.squeeze_shape(data.shape().to_vec())?;
        Ok(vec![Tensor::from(data.clone().into_shape(shape)?).into()])
    }

    /// Infers properties about the output tensors from the input tensors.
    fn infer_forward(&self, inputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>> {
        if inputs.len() != 1 {
            bail!("Squeeze operation only supports one input.");
        }

        if let Some(output) = infer_forward_concrete(self, &inputs)? {
            return Ok(Some(output));
        }

        // We can't say anything interesting if there are unknown dimensions,
        // because they could turn out to be Only(1), and so Squeeze would
        // have to remove them.
        let shape = match inputs[0].shape.concretize() {
            Some(shape) => self.squeeze_shape(shape)?.iter().collect(),
            None => shapefact![..],
        };

        let output = TensorFact {
            datatype: inputs[0].datatype,
            shape,
            value: valuefact!(_),
        };

        Ok(Some(vec![output]))
    }

    /// Returns the attributes of the operation and their values.
    fn get_attributes(&self) -> HashMap<&'static str, Attr> {
        hashmap!{
            "dims" => Attr::IsizeVec(&self.dims)
        }
    }

    /// Infers properties about the input tensors from the output tensors.
    fn infer_backward(&self, outputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>> {
        if outputs.len() < 1 {
            bail!("Squeeze operation only supports one output.");
        }

        Ok(Some(vec![TensorFact {
            datatype: outputs[0].datatype,
            shape: shapefact![..],
            value: valuefact!(_),
        }]))
    }
}
