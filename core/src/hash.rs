use std::hash::Hash;
use crate::ops::*;
use tract_linalg::hash::DynHash;

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

impl Hash for Box<dyn TypedOp> {
    fn hash<H: std::hash::Hasher>(&self, mut state: &mut H) {
        std::hash::Hash::hash(&self.type_id(), state);
        DynHash::dyn_hash(self, &mut state)
    }
}

impl Hash for Box<dyn PulsedOp> {
    fn hash<H: std::hash::Hasher>(&self, mut state: &mut H) {
        std::hash::Hash::hash(&self.type_id(), state);
        DynHash::dyn_hash(self, &mut state)
    }
}

impl<'a> Hash for &'a dyn PulsedOp {
    fn hash<H: std::hash::Hasher>(&self, mut state: &mut H) {
        std::hash::Hash::hash(&self.type_id(), state);
        DynHash::dyn_hash(self, &mut state)
    }
}

/*
pub fn hash_f32<H: std::hash::Hasher>(s: &f32, state: &mut H) {
    Hash::hash(&s.to_bits(), state)
}

struct WrappedHasher<'a>(&'a mut dyn std::hash::Hasher);

impl<'a> std::hash::Hasher for WrappedHasher<'a> {
    fn finish(&self) -> u64 {
        self.0.finish()
    }
    fn write(&mut self, bytes: &[u8]) {
        self.0.write(bytes)
    }
}

impl<O: std::hash::Hash> DynHash for O {
    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        self.hash(&mut WrappedHasher(state))
    }
}
*/
