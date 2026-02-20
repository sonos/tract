#![cfg(test)]
use tract_core::internal::*;

mod default {
    use super::*;
    pub fn default() -> &'static DefaultRuntime {
        &DefaultRuntime
    }
    include!(concat!(env!("OUT_DIR"), "/tests/default.rs"));
}

mod unoptimized {
    use super::*;

    pub fn unoptimized() -> &'static UnoptimizedRuntime {
        &UnoptimizedRuntime
    }

    #[derive(Debug)]
    pub struct UnoptimizedRuntime;

    impl Runtime for UnoptimizedRuntime {
        fn name(&self) -> StaticName {
            Cow::Borrowed("unoptimized")
        }
        fn prepare_with_options(
            &self,
            model: TypedModel,
            options: &RunOptions,
        ) -> TractResult<Box<dyn Runnable>> {
            Ok(Box::new(model.into_runnable_with_options(options)?))
        }
        fn check(&self) -> TractResult<()> {
            Ok(())
        }
    }

    include!(concat!(env!("OUT_DIR"), "/tests/unoptimized.rs"));
}
