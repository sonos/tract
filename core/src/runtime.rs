use std::fmt::Debug;

use crate::internal::*;

pub trait Runtime: Debug {
    fn name(&self) -> StaticName;
    fn prepare(&self, model: TypedModel) -> TractResult<Box<dyn Runnable>>;
}

pub trait Runnable: Debug {
    fn run(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        self.spawn()?.run(inputs)
    }
    fn spawn(&self) -> TractResult<Box<dyn State>>;
}

pub trait State {
    fn run(&mut self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>>;
}

#[derive(Debug)]
pub struct DefaultRuntime;

impl Runtime for DefaultRuntime {
    fn name(&self) -> StaticName {
        Cow::Borrowed("default")
    }

    fn prepare(&self, model: TypedModel) -> TractResult<Box<dyn Runnable>> {
        Ok(Box::new(model.into_optimized()?.into_runnable()?))
    }
}

impl Runnable for Arc<TypedRunnableModel> {
    fn spawn(&self) -> TractResult<Box<dyn State>> {
        Ok(Box::new(self.spawn()?))
    }
}

impl State for TypedSimpleState {
    fn run(&mut self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        self.run(inputs)
    }
}
