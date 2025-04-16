pub mod ops;
mod rewriter;
use rewriter::*;
use tract_nnef::internal::*;
use tract_nnef::tract_core::transform::ModelTransform;

pub fn get_transform(name: &str) -> Option<Box<dyn ModelTransform>> {
    match name {
        "detect-rms-norm" => Some(Box::new(RmsNormTransform)),
        "detect-apply-rope" => Some(Box::new(ApplyRopeTransform)),
        "detect-silu" => Some(Box::new(SiluTransform)),
        "detect-scaled-masked-softmax" => Some(Box::new(ScaledMaskedSoftmaxTransform)),
        "detect-gelu-approx" => Some(Box::new(GeluTransform)),
        "transformers-detect-all" => Some(Box::new(TransformersTransform)),
        _ => None,
    }
}

pub fn register(registry: &mut Registry) {
    registry.transforms = Box::new(|s| Ok(get_transform(s)));

    ops::rms_norm::register(registry);
    ops::silu::register(registry);
    ops::gelu_approximate::register(registry);
    ops::apply_rope::register(registry);
    ops::scaled_masked_softmax::register(registry);
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
