use tract_core::internal::*;

#[derive(Default, Debug, Copy, Clone)]
pub struct UnoptimizedRuntime;

register_runtime!(UnoptimizedRuntime = UnoptimizedRuntime);

impl Runtime for UnoptimizedRuntime {
    fn name(&self) -> StaticName {
        Cow::Borrowed("unoptimized")
    }

    fn prepare(&self, model: TypedModel) -> TractResult<Box<dyn Runnable>> {
        Ok(Box::new(SimplePlan::new(model)?))
    }
}
