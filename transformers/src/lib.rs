use tract_nnef::internal::*;

pub trait WithTractTransformers {
    fn enable_tract_transformers(&mut self);
    fn with_tract_transformers(self) -> Self;
}

impl WithTractTransformers for tract_nnef::framework::Nnef {
    fn enable_tract_transformers(&mut self) {
        self.enable_tract_core();
        self.registries.push(tract_transformers_registry());
    }

    fn with_tract_transformers(mut self) -> Self {
        self.enable_tract_transformers();
        self
    }
}

pub fn tract_transformers_registry() -> Registry {
    let reg = Registry::new("tract_transformers");
    reg
}

pub fn register_pulsifiers() {
    let _ = tract_transformers_registry();
}
