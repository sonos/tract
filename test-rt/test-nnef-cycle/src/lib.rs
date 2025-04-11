#![cfg(test)]
use std::fmt::Debug;

use log::*;
use tract_nnef::internal::*;
use tract_onnx_opl::*;

#[path = "../suite.rs"]
mod suite;

mod nnef_predump {
    use super::*;

    #[allow(dead_code)]
    struct NnefPredumpRuntime(Nnef);

    impl Debug for NnefPredumpRuntime {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "NnefPredumpRuntime")
        }
    }

    impl Runtime for NnefPredumpRuntime {
        fn name(&self) -> Cow<str> {
            "nnef_predump".into()
        }

        fn prepare(&self, model: TypedModel) -> TractResult<Box<dyn Runnable>> {
            let mut model = model.clone();
            tract_nnef::ser::rewrite_model(&mut model)?;
            Ok(Box::new(Arc::new(model.into_optimized()?.into_runnable()?)))
        }
    }

    fn runtime() -> &'static NnefPredumpRuntime {
        lazy_static::lazy_static! {
            static ref RT: NnefPredumpRuntime = NnefPredumpRuntime(tract_nnef::nnef().with_onnx());
        };
        &RT
    }

    include!(concat!(env!("OUT_DIR"), "/tests/nnef_cycle.rs"));
}

mod nnef_cycle {
    use tract_transformers::WithTractTransformers;

    use super::*;

    struct NnefCyclingRuntime(Nnef);

    impl Debug for NnefCyclingRuntime {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "NnefCyclingRuntime")
        }
    }

    impl Runtime for NnefCyclingRuntime {
        fn name(&self) -> Cow<str> {
            "nnef_cycle".into()
        }

        fn prepare(&self, model: TypedModel) -> TractResult<Box<dyn Runnable>> {
            info!("Store to NNEF");
            let mut buffer = vec![];
            eprintln!("{model}");
            self.0.write_to_tar(&model, &mut buffer)?;
            info!("Reload from NNEF");
            let reloaded = self.0.model_for_read(&mut &*buffer)?;
            // eprintln!("{}", reloaded.clone().into_decluttered().unwrap());
            Ok(Box::new(Arc::new(reloaded.into_optimized()?.into_runnable()?)))
        }
    }

    fn runtime() -> &'static NnefCyclingRuntime {
        lazy_static::lazy_static! {
            static ref RT: NnefCyclingRuntime = NnefCyclingRuntime(tract_nnef::nnef().with_onnx().with_tract_transformers());
        };
        &RT
    }

    include!(concat!(env!("OUT_DIR"), "/tests/nnef_cycle.rs"));
}
