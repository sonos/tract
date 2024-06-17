#![cfg(test)]

#[path = "../suite.rs"]
mod suite;

mod run_with_metal {
    use super::*;
    use tract_core::internal::*;
    use tract_core::transform::ModelTransform;

    #[derive(Debug)]
    struct RunWithMetal;

    impl Runtime for RunWithMetal {
        fn name(&self) -> Cow<str> {
            "run_with_metal".into()
        }

        fn prepare(&self, model: TypedModel) -> TractResult<Box<dyn Runnable>> {
            let metal_model = tract_metal::transform::MetalTransform.transform_into(&model)?;
            Ok(Box::new(Arc::new(metal_model.into_optimized()?.into_runnable()?)))
        }
    }

    fn runtime() -> &'static RunWithMetal {
        lazy_static::lazy_static! {
            static ref RT: RunWithMetal = RunWithMetal;
        };
        &RT
    }

    include!(concat!(env!("OUT_DIR"), "/tests/tests.rs"));
}
