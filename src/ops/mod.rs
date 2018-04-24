//! TensorFlow Ops

use std::fmt::Debug;
use std::collections::HashMap;
use std::sync::Arc;

use {Matrix, Result};

#[macro_use]
mod macros;

mod array;
mod math;
mod cast;
pub mod nn;
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

impl<M> From<M> for Input where Matrix: From<M> {
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
    fn eval(&self, inputs: Vec<Input>) -> Result<Vec<Input>>;
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
    fn eval(&self, _inputs: Vec<Input>) -> Result<Vec<Input>> {
        Err(format!("unimplemented operation: {}", self.0))?
    }
}

#[cfg(all(test, feature = "tensorflow"))]
pub mod proptests {
    #![allow(non_snake_case)]
    use tfpb;
    use tfpb::types::DataType;
    use tfpb::tensor_shape::TensorShapeProto;

    pub fn placeholder<Shape: Into<Option<TensorShapeProto>>>(name: &str, t: DataType, shape:Shape) -> tfpb::node_def::NodeDef {
        let mut node = tfpb::node().name(name).op("Placeholder").attr("dtype", t);
        if let Some(shape) = shape.into() {
            node = node.attr("shape", shape)
        }
        node
    }

    pub fn tensor_shape(dims: &[usize]) -> TensorShapeProto {
        use tfpb::tensor_shape::*;
        let mut shape = TensorShapeProto::new();
        shape.set_dim(dims.iter().map(|&d| {
            let mut dim = TensorShapeProto_Dim::new();
            dim.set_size(d as i64);
            dim
        }).collect());
        shape
    }


    pub fn placeholder_f32(name: &str) -> tfpb::node_def::NodeDef {
        placeholder(name, DataType::DT_FLOAT, None)
    }

    pub fn placeholder_i32(name: &str) -> tfpb::node_def::NodeDef {
        placeholder(name, DataType::DT_INT32, None)
    }

    pub fn compare<S:AsRef<str>>(
        graph: &[u8],
        inputs: Vec<(S, ::ops::Matrix)>,
        output: &str,
    ) -> Result<(), ::proptest::test_runner::TestCaseError> {
        let owned_names:Vec<String> = inputs.iter().map(|s| s.0.as_ref().to_string()).collect();
        let inputs:Vec<(&str, ::ops::Matrix)> = inputs.into_iter().zip(owned_names.iter()).map(|((_,m),s)| (&**s, m)).collect();
        let expected = ::tf::for_slice(&graph)?.run(inputs.clone(), output)?;
        let found = ::Model::for_reader(&*graph)?.run_with_names(inputs, output)?;
        prop_assert!(
            expected[0].shape() == found[0].shape() && expected[0].close_enough(&found[0]),
            "expected: {:?} found: {:?}",
            expected,
            found
        );
        Ok(())
    }

}
