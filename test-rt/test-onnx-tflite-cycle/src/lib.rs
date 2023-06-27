#![cfg(test)]
use std::borrow::Cow;

use log::*;
use tract_tflite::{internal::*, Tflite};

struct TfliteCyclingRuntime(Tflite);

impl Runtime for TfliteCyclingRuntime {
    fn name(&self) -> Cow<str> {
        "nnef_cycle".into()
    }

    fn prepare(&self, model: TypedModel) -> TractResult<Box<dyn Runnable>> {
        info!("Store to Tflite");
        let mut buffer = vec![];
        self.0.write(&model, &mut buffer)?;
        info!("Reload from Tflite");
        let reloaded = self.0.model_for_read(&mut &*buffer)?;
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
