use crate::ops::*;
use std::hash::Hash;

/*
impl Hash for Box<dyn Op> {
    fn hash<H: std::hash::Hasher>(&self, mut state: &mut H) {
        std::hash::Hash::hash(&self.type_id(), state);
        DynHash::dyn_hash(self, &mut state)
    }
}

impl<'a> Hash for &'a dyn Op {
    fn hash<H: std::hash::Hasher>(&self, mut state: &mut H) {
        std::hash::Hash::hash(&self.type_id(), state);
        DynHash::dyn_hash(self, &mut state)
    }
}
*/

impl Hash for Box<dyn TypedOp> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&self.type_id(), state);
        self.dyn_hash(state)
    }
}

/*
impl Hash for Box<dyn PulsedOp> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&self.type_id(), state);
        self.dyn_hash(state)
    }
}

impl<'a> Hash for &'a dyn PulsedOp {
    fn hash<H: std::hash::Hasher>(&self, mut state: &mut H) {
        std::hash::Hash::hash(&self.type_id(), state);
        DynHash::dyn_hash(self, &mut state)
    }
}
*/
