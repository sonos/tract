#![cfg(test)]
use log::*;
use tract_nnef::internal::*;
use tract_onnx_opl::*;

struct NnefCyclingRuntime(Nnef);

impl Runtime for NnefCyclingRuntime {
    fn name(&self) -> Cow<str> {
        "nnef_cycle".into()
    }

    fn prepare(&self, model: TypedModel) -> TractResult<Box<dyn Runnable>> {
        info!("Store to NNEF");
        let mut buffer = vec![];
        self.0.write_to_tar(&model, &mut buffer)?;
        info!("Reload from NNEF");
        let reloaded = self.0.model_for_read(&mut &*buffer)?;
        Ok(Box::new(reloaded.into_optimized()?.into_runnable()?))
    }
}

fn nnef_cycle() -> &'static NnefCyclingRuntime {
    lazy_static::lazy_static! {
        static ref RT: NnefCyclingRuntime = NnefCyclingRuntime(tract_nnef::nnef().with_onnx());
    };
    &RT
}

include!(concat!(env!("OUT_DIR"), "/tests/nnef_cycle.rs"));
