use std::collections::VecDeque;
use std::{fmt, ops};

use downcast_rs::Downcast;

use ops::prelude::*;

/// A streaming buffer for a Tensorflow operation.
///
/// This is used during streaming evaluation of models. Each node is given
/// a mutable reference to a buffer which it can use to store intermediary
/// results between evaluation steps. Every operation must provide its own
/// buffer type (or use one of the general ones defined below), which must
/// implement the OpBuffer trait. It should return a new instance of it in
/// the `Op::new_buffer` method, and downcast it from OpBuffer in `step`.
pub trait OpBuffer: Downcast + fmt::Debug + ::objekt::Clone + Send + 'static {}
clone_trait_object!(OpBuffer);
impl_downcast!(OpBuffer);

/// An empty buffer for operations which don't need one.
#[derive(Debug, Clone)]
pub struct EmptyBuffer {}

impl OpBuffer for EmptyBuffer {}

/// A buffer with a variable number of Value queues.
#[derive(Debug, Clone)]
pub struct QueuesBuffer(TVec<VecDeque<Value>>);

impl OpBuffer for QueuesBuffer {}

impl QueuesBuffer {
    /// Creates a new buffer with a given number of queues.
    pub fn new(size: usize) -> QueuesBuffer {
        QueuesBuffer(tvec![VecDeque::new(); size])
    }

    /// Appends a new Value to each queue in the buffer.
    pub fn append(&mut self, views: TVec<StepValue>) -> TfdResult<()> {
        if views.len() > self.0.len() {
            bail!("There are more input Values than queues in the buffer.");
        }

        for (i, view) in views.into_iter().enumerate() {
            if let Some(v) = view.into_value() {
                self.0[i].push_back(v);
            }
        }

        Ok(())
    }

    /// Returns an iterator over all the queues in the buffer.
    pub fn iter<'a>(&'a mut self) -> impl Iterator<Item = &'a VecDeque<Value>> {
        self.0.iter()
    }

    /// Returns a mutable iterator over all the queues in the buffer.
    pub fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut VecDeque<Value>> {
        self.0.iter_mut()
    }
}

impl ops::Index<usize> for QueuesBuffer {
    type Output = VecDeque<Value>;

    fn index(&self, index: usize) -> &VecDeque<Value> {
        &self.0[index]
    }
}

impl ops::IndexMut<usize> for QueuesBuffer {
    fn index_mut(&mut self, index: usize) -> &mut VecDeque<Value> {
        &mut self.0[index]
    }
}
