use tract_nnef::internal::*;

mod exp_unit_norm;

pub trait WithExtra {
    fn enable_extra(&mut self);
    fn with_extra(self) -> Self;
}

impl WithExtra for tract_nnef::framework::Nnef {
    fn enable_extra(&mut self) {
        self.enable_tract_core();
        self.registries.push(tract_nnef_registry());
    }

    fn with_extra(mut self) -> Self {
        self.enable_extra();
        self
    }
}

pub fn tract_nnef_registry() -> Registry {
    let mut reg = Registry::new("tract_extra");
    exp_unit_norm::register(&mut reg);
    reg
}

pub fn register_pulsifiers() {
    let _ = tract_nnef_registry();
}
