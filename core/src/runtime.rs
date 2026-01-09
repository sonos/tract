use std::fmt::Debug;

use crate::internal::*;

pub trait Runtime: Debug + Send + Sync + 'static {
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

pub struct InventorizedRuntime(pub &'static dyn Runtime);

impl Runtime for InventorizedRuntime {
    fn name(&self) -> StaticName {
        self.0.name()
    }

    fn prepare(&self, model: TypedModel) -> TractResult<Box<dyn Runnable>> {
        self.0.prepare(model)
    }
}

impl Debug for InventorizedRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

inventory::collect!(InventorizedRuntime);

pub fn runtimes() -> impl Iterator<Item = &'static dyn Runtime> {
    inventory::iter::<InventorizedRuntime>().map(|ir| ir.0)
}

#[macro_export]
macro_rules! register_runtime {
    ($type: ty= $val:expr) => {
        static D: $type = $val;
        inventory::submit! { $crate::runtime::InventorizedRuntime(&D) }
    };
}

register_runtime!(DefaultRuntime = DefaultRuntime);
