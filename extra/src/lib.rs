use tract_nnef::internal::*;

mod exp_unit_norm;

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
    reg
}

pub fn register_pulsifiers() {
    let _ = tract_extra_registry();
}
