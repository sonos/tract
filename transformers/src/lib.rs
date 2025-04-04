mod rewriter;
mod rewrite_rules;
pub mod ops;
use rewriter::*;
use tract_nnef::internal::*;
use tract_nnef::tract_core::transform::ModelTransform;

pub fn get_transform(name: &str) -> Option<Box<dyn ModelTransform>> {
    match name {
        "as-rms-norm" => Some(Box::new(RmsNormTransform)),
        "as-apply-rope" => Some(Box::new(ApplyRopeTransform)),
        "as-silu" => Some(Box::new(SiluTransform)),
        _ => None,
    }
}
 
pub fn register(registry: &mut Registry) {
    registry.transforms = Box::new(|s| Ok(get_transform(s)));

    ops::rms_norm::register(registry);
}

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
    let mut reg = Registry::new("tract_transformers")
        .with_doc("Extension `tract_transformers` extends NNEF with operators")
        .with_doc("for transformer networks.")
        .with_doc("")
        .with_doc("Add `extension tract_transformers` to `graph.nnef`");
    register(&mut reg);
    reg
}

pub fn register_pulsifiers() {
    let _ = tract_transformers_registry();
}
