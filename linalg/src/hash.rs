use std::hash::{Hash, Hasher};

pub trait SloppyHash {
    fn sloppy_hash<S: Hasher>(&self, state: &mut S);
}

impl SloppyHash for crate::f16::f16 {
    fn sloppy_hash<S: Hasher>(&self, state: &mut S) {
        unsafe { std::mem::transmute_copy::<crate::f16::f16, i16>(self).hash(state) }
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

#[macro_export]
macro_rules! impl_dyn_hash {
    ($t: ty) => {
        impl DynHash for $t {
            fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
                $crate::hash::dyn_hash(self, state)
            }
        }
    };
}

pub trait DynHash {
    fn dyn_hash(&self, state: &mut dyn Hasher);
}

pub fn hash_f32<H: Hasher>(s: &f32, state: &mut H) {
    Hash::hash(&s.to_bits(), state)
}

pub fn hash_opt_f32<H: Hasher>(s: &Option<f32>, state: &mut H) {
    Hash::hash(&s.is_some(), state);
    if let Some(s) = s {
        Hash::hash(&s.to_bits(), state)
    }
}

struct WrappedHasher<'a>(&'a mut dyn Hasher);

impl<'a> Hasher for WrappedHasher<'a> {
    fn finish(&self) -> u64 {
        self.0.finish()
    }
    fn write(&mut self, bytes: &[u8]) {
        self.0.write(bytes)
    }
}

pub fn dyn_hash<H: Hash>(h: H, s: &mut dyn Hasher) {
    h.hash(&mut WrappedHasher(s))
}
