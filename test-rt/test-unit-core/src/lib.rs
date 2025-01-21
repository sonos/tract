#![cfg(test)]
use tract_core::internal::*;

mod raw {
    use super::*;

    pub fn raw() -> &'static RawRuntime {
        &RawRuntime
    }

    #[derive(Debug)]
    pub struct RawRuntime;

    impl Runtime for RawRuntime {
        fn name(&self) -> Cow<str> {
            Cow::Borrowed("raw")
        }
        fn prepare(&self, model: TypedModel) -> TractResult<Box<dyn Runnable>> {
            Ok(Box::new(Arc::new(model.into_runnable()?)))
        }
    }

    include!(concat!(env!("OUT_DIR"), "/tests/raw.rs"));
}

mod decluttered {
    use super::*;

    pub fn decluttered() -> &'static DeclutteredRuntime {
        &DeclutteredRuntime
    }

    #[derive(Debug)]
    pub struct DeclutteredRuntime;

    impl Runtime for DeclutteredRuntime {
        fn name(&self) -> Cow<str> {
            Cow::Borrowed("decluttered")
        }
        fn prepare(&self, model: TypedModel) -> TractResult<Box<dyn Runnable>> {
            Ok(Box::new(Arc::new(model.into_decluttered()?.into_runnable()?)))
        }
    }

    include!(concat!(env!("OUT_DIR"), "/tests/decluttered.rs"));
}

mod optimized {
    use super::*;

    pub fn optimized() -> &'static DefaultRuntime {
        &DefaultRuntime
    }
    include!(concat!(env!("OUT_DIR"), "/tests/optimized.rs"));
}
