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
        fn name(&self) -> Cow<str> {
            Cow::Borrowed("unoptimized")
        }
        fn prepare(&self, model: TypedModel) -> TractResult<Box<dyn Runnable>> {
            Ok(Box::new(Arc::new(model.into_runnable()?)))
        }
    }

    include!(concat!(env!("OUT_DIR"), "/tests/unoptimized.rs"));
}

