mod category_mapper;
mod tree_ensemble_classifier;

use crate::model::OnnxOpRegister;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    category_mapper::register_all_ops(reg);
    tree_ensemble_classifier::register_all_ops(reg);
}
