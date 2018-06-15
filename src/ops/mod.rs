//! TensorFlow Ops
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;
#[cfg(feature = "serialize")]
use std::result::Result as StdResult;

use {Result, Tensor};
use analyser::TensorFact;
use tfpb::types::DataType;

#[cfg(feature = "serialize")]
use serde::ser::{Serialize, Serializer};

#[macro_use]
mod macros;

mod array;
mod cast;
#[cfg(features = "image_ops")]
pub mod image;
pub mod konst;
mod math;
pub mod nn;

#[derive(Debug, Clone)]
pub enum TensorView {
    Owned(Tensor),
    Shared(Arc<Tensor>),
}

impl TensorView {
    pub fn into_tensor(self) -> Tensor {
        match self {
            TensorView::Owned(m) => m,
            TensorView::Shared(m) => m.as_ref().clone(),
        }
    }
    pub fn as_tensor(&self) -> &Tensor {
        match self {
            &TensorView::Owned(ref m) => &m,
            &TensorView::Shared(ref m) => m.as_ref(),
        }
    }
}

impl<M> From<M> for TensorView
where
    Tensor: From<M>,
{
    fn from(m: M) -> TensorView {
        TensorView::Owned(m.into())
    }
}

impl From<Arc<Tensor>> for TensorView {
    fn from(m: Arc<Tensor>) -> TensorView {
        TensorView::Shared(m)
    }
}

impl ::std::ops::Deref for TensorView {
    type Target = Tensor;
    fn deref(&self) -> &Tensor {
        match self {
            &TensorView::Owned(ref m) => &m,
            &TensorView::Shared(ref m) => m.as_ref(),
        }
    }
}

impl PartialEq for TensorView {
    fn eq(&self, other: &TensorView) -> bool {
        self.as_tensor() == other.as_tensor()
    }
}

// TODO(liautaud): Find a more generic way to do this.
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[derive(Debug)]
pub enum Attr<'a> {
    I64(i64),
    Usize(usize),
    DataType(DataType),
    Tensor(&'a Tensor),
    IsizeVec(&'a Vec<isize>),
}

pub trait Op: Debug + Send + Sync + 'static {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: Vec<TensorView>) -> Result<Vec<TensorView>>;

    /// Returns the attributes of the operation and their values.
    fn get_attributes(&self) -> HashMap<&'static str, Attr>;

    /// Infers properties about the output tensors from the input tensors.
    /// Returns Err in case of an unrecoverable error during the inference,
    /// Ok(None) if there was nothing to infer, and Ok(Some(_)) otherwise.
    fn infer_forward(&self, _inputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>>;

    /// Infers properties about the input tensors from the output tensors.
    /// Returns Err in case of an unrecoverable error during the inference,
    /// Ok(None) if there was nothing to infer, and Ok(Some(_)) otherwise.
    fn infer_backward(&self, _outputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>>;
}

#[cfg(feature = "serialize")]
impl Serialize for Op {
    fn serialize<S>(&self, serializer: S) -> StdResult<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.get_attributes().serialize(serializer)
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
    fn eval(&self, _inputs: Vec<TensorView>) -> Result<Vec<TensorView>> {
        Err(format!("unimplemented operation: {}", self.0))?
    }

    /// Returns the attributes of the operation and their values.
    fn get_attributes(&self) -> HashMap<&'static str, Attr> {
        unimplemented!()
    }

    /// Infers properties about the output tensors from the input tensors.
    fn infer_forward(&self, _inputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>> {
        unimplemented!()
    }

    /// Infers properties about the input tensors from the output tensors.
    fn infer_backward(&self, _outputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>> {
        unimplemented!()
    }
}
