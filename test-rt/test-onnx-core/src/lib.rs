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

mod as_blas {
    use super::*;

    pub fn as_blas() -> &'static AsBlasRuntime {
        &AsBlasRuntime
    }

    #[derive(Debug)]
    pub struct AsBlasRuntime;

    impl Runtime for AsBlasRuntime {
        fn name(&self) -> Cow<str> {
            Cow::Borrowed("as_blas")
        }
        fn prepare(&self, mut model: TypedModel) -> TractResult<Box<dyn Runnable>> {
            tract_core::transform::get_transformer("as-blas").unwrap().transform(&mut model)?;
            Ok(Box::new(Arc::new(model.into_runnable()?)))
        }
    }

    include!(concat!(env!("OUT_DIR"), "/tests/as_blas.rs"));
}
