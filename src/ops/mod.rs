//! TensorFlow Ops
use std::collections::HashMap;
use std::collections::VecDeque;
use std::fmt::Debug;
#[cfg(feature = "serialize")]
use std::result::Result as StdResult;
use std::sync::Arc;
use std::mem;

use analyser::TensorFact;
use tfpb::types::DataType;
use {Result, Tensor};

#[cfg(feature = "serialize")]
use serde::ser::{Serialize, Serializer};
use objekt;

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
    /// Creates a shared TensorView from any TensorView.
    pub fn into_shared(self) -> TensorView {
        match self {
            TensorView::Owned(m) => TensorView::Shared(Arc::new(m)),
            TensorView::Shared(_) => self,
        }
    }

    /// Creates a Tensor from a TensorView.
    pub fn into_tensor(self) -> Tensor {
        match self {
            TensorView::Owned(m) => m,
            TensorView::Shared(m) => m.as_ref().clone(),
        }
    }

    /// Returns a reference to the Tensor wrapped inside a TensorView.
    pub fn as_tensor(&self) -> &Tensor {
        match self {
            &TensorView::Owned(ref m) => &m,
            &TensorView::Shared(ref m) => m.as_ref(),
        }
    }

    /// Returns a shared copy of the TensorView, turning the one passed
    /// as argument into a TensorView::Shared if necessary.
    pub fn share(&mut self) -> TensorView {
        // This is somewhat ugly, but sadly we couldn't find any other
        // way to implement it. If we try to write something like:
        //   *self = TensorView::Shared(Arc::new(*m))
        // the borrow checker will complain about *m being moved out of
        // borrowed content, which makes sense but doesn't apply in our
        // case because we will "give m back" to the TensorView, except
        // wrapped around an Arc. The only way to get ownership of m is
        // to use mem::replace, which means we have to create a "dummy"
        // value to replace self first.
        if let TensorView::Owned(_) = self {
            let dummy = TensorView::Owned(Tensor::i32s(&[], &[0]).unwrap());
            let shared = match mem::replace(self, dummy) {
                TensorView::Owned(m) => TensorView::Shared(Arc::new(m)),
                _ => panic!()
            };

            *self = shared;
        }

        self.clone()
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

pub trait Op: Debug + objekt::Clone + Send + Sync + 'static {
    /// Returns the attributes of the operation and their values.
    fn get_attributes(&self) -> HashMap<&'static str, Attr>;

    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: Vec<TensorView>) -> Result<Vec<TensorView>>;

    /// Evaluates one step of the operation on the given input tensors.
    /// This is only implemented for operators which support streaming.
    ///
    /// The input tensors are annotated with an Option<usize>:
    /// - None if the tensor doesn't have a streaming dimension.
    /// - Option(d) if the tensor is being streamed on dimension d.
    ///
    /// If an input tensor has a streaming dimension, the corresponding
    /// TensorView will only contain a _chunk_ of input of size 1 along
    /// that dimension. Note that each chunk will only be passed once
    /// to the step function, so it should use the provided buffer to
    /// store whichever chunks it needs for future computations.
    ///
    /// The function should return Some(chunks) when it has computed
    /// new chunks, and None if it has computed an intermediary result
    /// successfully but doesn't have new output chunks ready yet.
    ///
    /// For operators like Concat, multiple input tensors might have a
    /// streaming dimension. In that case, at each call to step, only
    /// one of the streaming inputs will receive new chunk while the
    /// others will receive None.
    fn step(
        &self,
        _inputs: Vec<(Option<usize>, Option<TensorView>)>,
        _buffer: &mut Vec<VecDeque<TensorView>>,
    ) -> Result<Option<Vec<TensorView>>> {
        bail!("Streaming is not available for this operator.")
    }

    /// Infers properties about the output tensors from the input tensors.
    /// Returns Err in case of an unrecoverable error during the inference,
    /// Ok(None) if there was nothing to infer, and Ok(Some(_)) otherwise.
    fn infer_forward(&self, _inputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>>;

    /// Infers properties about the input tensors from the output tensors.
    /// Returns Err in case of an unrecoverable error during the inference,
    /// Ok(None) if there was nothing to infer, and Ok(Some(_)) otherwise.
    fn infer_backward(&self, _outputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>>;
}

clone_trait_object!(Op);

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

#[derive(Debug, Clone)]
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
