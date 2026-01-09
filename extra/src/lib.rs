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
    is_nan::register(&mut reg);
    reg.register_element_wise(
        "tract_extra_is_inf",
        TypeId::of::<is_inf::IsInf>(),
        Box::new(is_inf::dump),
        is_inf::parameters(),
        is_inf::load,
    );
    reg
}

pub fn register_pulsifiers() {
    let _ = tract_extra_registry();
}
