#[macro_use]
extern crate educe;

use tract_nnef::internal::*;

#[macro_use]
mod macros;

pub mod einsum;
pub mod erf;
pub mod is_inf;
pub mod is_nan;
pub mod lrn;
pub mod ml;

pub trait WithOnnx {
    fn with_onnx(self) -> Self;
}

impl WithOnnx for tract_nnef::framework::Nnef {
    fn with_onnx(mut self) -> Self {
        self = self.with_tract_core();
        self.registries.push(onnx_opl_registry());
        self
    }
}

fn onnx_opl_registry() -> Registry {
    let mut registry: Registry = Registry::new("tract_onnx");
    ml::register(&mut registry);
    registry.register_unit_element_wise("tract_onnx_erf", &erf::Erf {});
    registry.register_element_wise(
        "tract_onnx_isinf",
        TypeId::of::<is_inf::IsInf>(),
        is_inf::dump,
        is_inf::parameters(),
        is_inf::load,
    );
    registry.register_unit_element_wise("tract_onnx_is_nan", &is_nan::IsNan {});
    registry.register_dumper(TypeId::of::<lrn::Lrn>(), lrn::dump);
    registry.register_primitive("tract_onnx_lrn", &lrn::parameters(), lrn::load);
    registry
}
