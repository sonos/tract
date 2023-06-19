use crate::internal::*;

pub trait Runnable {
    fn run(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>>;
}

pub trait Runtime {
    fn name(&self) -> Cow<str>;
    fn prepare(&self, model: TypedModel) -> TractResult<Box<dyn Runnable>>;
}

impl Runnable for TypedRunnableModel<TypedModel> {
    fn run(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        SimplePlan::run(self, inputs)
    }
}

pub struct DefaultRuntime;

impl Runtime for DefaultRuntime {
    fn name(&self) -> Cow<str> {
        Cow::Borrowed("default")
    }
    fn prepare(&self, model: TypedModel) -> TractResult<Box<dyn Runnable>> {
        Ok(Box::new(model.into_optimized()?.into_runnable()?))
    }
}

