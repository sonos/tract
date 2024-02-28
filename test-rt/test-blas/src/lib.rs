#![cfg(test)]
use std::fmt::Debug;

use log::*;
use tract_core::internal::*;
use tract_onnx_opl::*;

#[path="../suite.rs"]
mod suite;

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

