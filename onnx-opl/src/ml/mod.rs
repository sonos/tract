use tract_nnef::internal::*;

pub mod category_mapper;
pub mod tree;
pub mod tree_ensemble_classifier;

pub use category_mapper::{DirectLookup, ReverseLookup};

pub fn register(registry: &mut Registry) {
    category_mapper::register(registry);
    tree_ensemble_classifier::register(registry);
}
