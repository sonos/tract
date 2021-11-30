use crate::ops::*;

use std::hash::{Hash, Hasher};

pub trait SloppyHash {
    fn sloppy_hash<S: Hasher>(&self, state: &mut S);
}

impl SloppyHash for tract_data::prelude::f16 {
    fn sloppy_hash<S: Hasher>(&self, state: &mut S) {
        unsafe { std::mem::transmute_copy::<tract_data::prelude::f16, i16>(self).hash(state) }
    }
}

impl SloppyHash for f32 {
    fn sloppy_hash<S: Hasher>(&self, state: &mut S) {
        self.to_bits().hash(state)
    }
}

impl SloppyHash for f64 {
    fn sloppy_hash<S: Hasher>(&self, state: &mut S) {
        self.to_bits().hash(state)
    }
}

macro_rules! impl_sloppy_hash {
    ($t: ty) => {
        impl SloppyHash for $t {
            fn sloppy_hash<S: Hasher>(&self, state: &mut S) {
                self.hash(state)
            }
        }
    };
}

impl_sloppy_hash!(bool);
impl_sloppy_hash!(i8);
impl_sloppy_hash!(i16);
impl_sloppy_hash!(i32);
impl_sloppy_hash!(i64);
impl_sloppy_hash!(u8);
impl_sloppy_hash!(u16);
impl_sloppy_hash!(u32);
impl_sloppy_hash!(u64);
impl_sloppy_hash!(String);

pub fn hash_f32<H: Hasher>(s: &f32, state: &mut H) {
    Hash::hash(&s.to_bits(), state)
}

pub fn hash_opt_f32<H: Hasher>(s: &Option<f32>, state: &mut H) {
    Hash::hash(&s.is_some(), state);
    if let Some(s) = s {
        Hash::hash(&s.to_bits(), state)
    }
}

impl Hash for Box<dyn TypedOp> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&self.type_id(), state);
        self.dyn_hash(state)
    }
}
