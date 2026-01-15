#![cfg(test)]
use std::fmt::Debug;

use tract_core::internal::*;

#[path = "../suite.rs"]
mod suite;

mod as_blas {
    use super::*;

    pub fn as_blas() -> &'static AsBlasRuntime {
        &AsBlasRuntime
    }

    #[derive(Debug)]
    pub struct AsBlasRuntime;

    impl Runtime for AsBlasRuntime {
        fn name(&self) -> StaticName {
            Cow::Borrowed("as_blas")
        }
        fn prepare_with_options(
            &self,
            mut model: TypedModel,
            options: &PlanOptions,
        ) -> TractResult<Box<dyn Runnable>> {
            tract_core::transform::get_transform("as-blas").unwrap().transform(&mut model)?;
            Ok(Box::new(model.into_runnable_with_options(options)?))
        }
    }

    include!(concat!(env!("OUT_DIR"), "/tests/as_blas.rs"));
}
