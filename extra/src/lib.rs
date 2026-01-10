use tract_nnef::internal::*;

mod exp_unit_norm;
pub mod is_inf;
pub mod is_nan;

pub trait WithTractExtra {
    fn enable_tract_extra(&mut self);
    fn with_tract_extra(self) -> Self;
}

impl WithTractExtra for tract_nnef::framework::Nnef {
    fn enable_tract_extra(&mut self) {
        self.enable_tract_core();
        self.registries.push(tract_extra_registry());
    }

    fn with_tract_extra(mut self) -> Self {
        self.enable_tract_extra();
        self
    }
}

pub fn tract_extra_registry() -> Registry {
    let mut reg = Registry::new("tract_extra");
    exp_unit_norm::register(&mut reg);
    register_shared_with_onnx_operators(&mut reg);
    reg
}

/// Register operators that are shared between `tract_extra` and `tract_onnx`.
pub fn register_shared_with_onnx_operators(reg: &mut Registry) {
    is_nan::register(reg);
    reg.register_element_wise(
        "tract_extra_is_inf",
        TypeId::of::<is_inf::IsInf>(),
        Box::new(is_inf::dump),
        is_inf::parameters(),
        is_inf::load,
    );
}

pub fn register_pulsifiers() {
    let _ = tract_extra_registry();
}
