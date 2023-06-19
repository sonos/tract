use tract_core::internal::*;

mod onnx;

pub fn default() -> &'static DefaultRuntime {
    &DefaultRuntime
}

pub struct UnoptimizedRuntime;

impl Runtime for UnoptimizedRuntime {
    fn name(&self) -> Cow<str> {
        Cow::Borrowed("unoptimized")
    }
    fn prepare(&self, model: TypedModel) -> TractResult<Box<dyn Runnable>> {
        Ok(Box::new(model.into_runnable()?))
    }
}

pub fn unoptimized() -> &'static UnoptimizedRuntime {
    &UnoptimizedRuntime
}

/*
pub fn nnef_cycling() -> &'static NnefCyclingRuntime {
    lazy_static! {
        static ref RT: NnefCyclingRuntime = NnefCyclingRuntime(tract_nnef::nnef().with_onnx());
    };
    &RT
}
*/

include!(concat!(env!("OUT_DIR"), "/tests/default.rs"));
include!(concat!(env!("OUT_DIR"), "/tests/unoptimized.rs"));
