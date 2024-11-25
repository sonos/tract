#![allow(clippy::len_zero)]

use tract_nnef::internal::*;

pub mod is_inf;
pub mod is_nan;
pub mod lrn;
pub mod ml;
pub mod multinomial;
pub mod non_max_suppression;
pub mod random;

pub trait WithOnnx {
    fn with_onnx(self) -> Self;
    fn enable_onnx(&mut self);
}

impl WithOnnx for tract_nnef::framework::Nnef {
    fn enable_onnx(&mut self) {
        self.enable_tract_core();
        self.registries.push(onnx_opl_registry());
    }
    fn with_onnx(mut self) -> Self {
        self.enable_onnx();
        self
    }
}

pub fn onnx_opl_registry() -> Registry {
    let mut registry: Registry = Registry::new("tract_onnx")
        .with_doc("Extension `tract_onnx` extends NNEF for supporting some corner case ONNX operators.")
        .with_doc("")
        .with_doc("Add `extension tract_onnx` to `graph.nnef`");
    ml::register(&mut registry);
    non_max_suppression::register(&mut registry);
    multinomial::register(&mut registry);
    random::register(&mut registry);
    registry.register_element_wise(
        "tract_onnx_isinf",
        TypeId::of::<is_inf::IsInf>(),
        Box::new(is_inf::dump),
        is_inf::parameters(),
        is_inf::load,
    );
    registry.register_unit_element_wise("tract_onnx_is_nan", &is_nan::IsNan {});
    registry.register_dumper(lrn::dump);
    registry.register_primitive(
        "tract_onnx_lrn",
        &lrn::parameters(),
        &[("output", TypeName::Scalar.tensor())],
        lrn::load,
    );
    registry
}
