//! TensorFlow Ops

use std::fmt::Debug;
use std::collections::HashMap;
use std::sync::Arc;

use analyser::{ATensor, AShape};

use {Matrix, Result};

#[macro_use]
mod macros;

mod array;
mod math;
mod cast;
pub mod nn;
#[cfg(features="image_ops")]
pub mod image;
pub mod konst;

#[derive(Debug, Clone)]
pub enum Input {
    Owned(Matrix),
    Shared(Arc<Matrix>),
}

impl Input {
    pub fn into_matrix(self) -> Matrix {
        match self {
            Input::Owned(m) => m,
            Input::Shared(m) => m.as_ref().clone(),
        }
    }
    pub fn as_matrix(&self) -> &Matrix {
        match self {
            &Input::Owned(ref m) => &m,
            &Input::Shared(ref m) => m.as_ref(),
        }
    }
}

impl<M> From<M> for Input
where
    Matrix: From<M>,
{
    fn from(m: M) -> Input {
        Input::Owned(m.into())
    }
}

impl From<Arc<Matrix>> for Input {
    fn from(m: Arc<Matrix>) -> Input {
        Input::Shared(m)
    }
}

impl ::std::ops::Deref for Input {
    type Target = Matrix;
    fn deref(&self) -> &Matrix {
        match self {
            &Input::Owned(ref m) => &m,
            &Input::Shared(ref m) => m.as_ref(),
        }
    }
}

impl PartialEq for Input {
    fn eq(&self, other: &Input) -> bool {
        self.as_matrix() == other.as_matrix()
    }
}

pub trait Op: Debug + Send + Sync + 'static {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: Vec<Input>) -> Result<Vec<Input>>;

    /// Infers properties about the output tensors from the input tensors.
    fn infer_forward(&self, inputs: Vec<&ATensor>) -> Result<Vec<ATensor>> {
        let input_shapes = inputs
            .iter()
            .map(|t| &t.shape)
            .collect::<Vec<_>>();

        let outputs = self
            .infer_shape_forward(input_shapes)?
            .into_iter()
            .enumerate()
            .map(|(i, s)| ATensor {
                datatype: inputs[i].datatype.clone(),
                value: inputs[i].value.clone(),
                shape: s,
            })
            .collect::<Vec<_>>();

        Ok(outputs)
    }

    /// Infers properties about the input tensors from the output tensors.
    fn infer_backward(&self, outputs: Vec<&ATensor>) -> Result<Vec<ATensor>> {
        let output_shapes = outputs
            .iter()
            .map(|t| &t.shape)
            .collect::<Vec<_>>();

        let inputs = self
            .infer_shape_backward(output_shapes)?
            .into_iter()
            .enumerate()
            .map(|(i, s)| ATensor {
                datatype: outputs[i].datatype.clone(),
                value: outputs[i].value.clone(),
                shape: s,
            })
            .collect::<Vec<_>>();

        Ok(inputs)
    }

    /// Infers properties about the shapes of the input tensors from the shapes
    /// of the output tensors.
    fn infer_shape_forward(&self, _input_shapes: Vec<&AShape>) -> Result<Vec<AShape>> {
        unimplemented!()
    }

    /// Infers properties about the shapes of the input tensors from the shapes
    /// of the output tensors.
    fn infer_shape_backward(&self, _output_shapes: Vec<&AShape>) -> Result<Vec<AShape>> {
        unimplemented!()
    }
}

type OpRegister = HashMap<&'static str, fn(&::tfpb::node_def::NodeDef) -> Result<Box<Op>>>;

pub struct OpBuilder(OpRegister);

impl OpBuilder {
    pub fn new() -> OpBuilder {
        let mut reg = OpRegister::new();
        array::register_all_ops(&mut reg);
        cast::register_all_ops(&mut reg);
        konst::register_all_ops(&mut reg);
        math::register_all_ops(&mut reg);
        nn::register_all_ops(&mut reg);
        OpBuilder(reg)
    }

    pub fn build(&self, pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        match self.0.get(pb.get_op()) {
            Some(builder) => builder(pb),
            None => Ok(Box::new(UnimplementedOp(
                pb.get_op().to_string(),
                pb.to_owned(),
            ))),
        }
    }
}

#[derive(Debug)]
pub struct UnimplementedOp(String, ::tfpb::node_def::NodeDef);

impl Op for UnimplementedOp {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, _inputs: Vec<Input>) -> Result<Vec<Input>> {
        Err(format!("unimplemented operation: {}", self.0))?
    }
}

