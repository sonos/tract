use std::fmt::Debug;

use crate::internal::*;

pub trait Runtime: Debug {
    fn name(&self) -> Cow<str>;
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
    fn name(&self) -> Cow<str> {
        Cow::Borrowed("default")
    }

    fn prepare(&self, model: TypedModel) -> TractResult<Box<dyn Runnable>> {
        Ok(Box::new(Arc::new(model.into_optimized()?.into_runnable()?)))
    }
}

impl Runnable for Arc<TypedRunnableModel<TypedModel>> {
    fn spawn(&self) -> TractResult<Box<dyn State>> {
        Ok(Box::new(SimpleState::new(self.clone())?))
    }
}

impl State for TypedSimpleState<TypedModel, Arc<TypedRunnableModel<TypedModel>>> {
    fn run(&mut self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        self.run(inputs)
    }
}
