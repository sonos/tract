#![cfg(test)]
use std::borrow::Cow;

use log::*;
use tract_tflite::internal::*;
use tract_tflite::Tflite;

#[path = "../suite.rs"]
mod suite;

mod tflite_predump {
    use super::*;
    struct TflitePredump(Tflite);

    impl Runtime for TflitePredump {
        fn name(&self) -> Cow<str> {
            "tflite-predump".into()
        }

        fn prepare(&self, model: TypedModel) -> TractResult<Box<dyn Runnable>> {
            let mut model = model.clone();
            tract_tflite::rewriter::rewrite_for_tflite(&mut model)?;
            Ok(Box::new(model.into_runnable()?))
        }
    }

    fn tflite_predump() -> &'static TflitePredump {
        lazy_static::lazy_static! {
            static ref RT: TflitePredump = TflitePredump(Tflite::default());
        };
        &RT
    }

    include!(concat!(env!("OUT_DIR"), "/tests/tflite_predump.rs"));
}

mod tflite_cycle {
    use super::*;
    struct TfliteCyclingRuntime(Tflite);

    impl Runtime for TfliteCyclingRuntime {
        fn name(&self) -> Cow<str> {
            "tflite_cycle".into()
        }

        fn prepare(&self, model: TypedModel) -> TractResult<Box<dyn Runnable>> {
            info!("Store to Tflite");
            let mut buffer = vec![];
            self.0.write(&model, &mut buffer)?;
            info!("Reload from Tflite");
            let reloaded = self.0.model_for_read(&mut &*buffer)?;
            println!("{reloaded:#?}");
            Ok(Box::new(reloaded.into_optimized()?.into_runnable()?))
        }
    }

    fn tflite_cycle() -> &'static TfliteCyclingRuntime {
        lazy_static::lazy_static! {
            static ref RT: TfliteCyclingRuntime = TfliteCyclingRuntime(Tflite::default());
        };
        &RT
    }

    include!(concat!(env!("OUT_DIR"), "/tests/tflite_cycle.rs"));
}

