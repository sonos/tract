use tract_nnef::internal::*;

pub mod category_mapper;
pub mod tree;
pub mod tree_ensemble_classifier;

pub fn register(registry: &mut Registry) {
    category_mapper::register(registry);
}
