use tract_core::internal::*;

#[derive(Default, Debug, Copy, Clone)]
pub struct UnoptimizedRuntime;

register_runtime!(UnoptimizedRuntime = UnoptimizedRuntime);

impl Runtime for UnoptimizedRuntime {
    fn name(&self) -> StaticName {
        Cow::Borrowed("unoptimized")
    }

    fn prepare_with_options(
        &self,
        model: TypedModel,
        options: &PlanOptions,
    ) -> TractResult<Box<dyn Runnable>> {
        Ok(Box::new(SimplePlan::new_with_options(model, options)?))
    }
}
