use crate::ops::*;
use std::hash::Hash;

impl Hash for Box<dyn TypedOp> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&self.type_id(), state);
        self.dyn_hash(state)
    }
}
