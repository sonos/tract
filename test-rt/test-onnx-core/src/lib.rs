#![cfg(test)]
mod default {
    use tract_core::internal::*;
    pub fn default() -> &'static DefaultRuntime {
        &DefaultRuntime
    }
    include!(concat!(env!("OUT_DIR"), "/tests/default.rs"));
}

mod unoptimized {
    use tract_core::internal::*;

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

