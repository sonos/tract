use std::hash::{Hash, Hasher};

#[macro_export]
macro_rules! impl_dyn_hash {
    ($t: ty) => {
        impl $crate::hash::DynHash for $t {
            fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
                $crate::hash::dyn_hash(self, state)
            }
        }
    };
}

pub trait DynHash {
    fn dyn_hash(&self, state: &mut dyn Hasher);
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

#[allow(dead_code)]
pub fn dyn_hash<H: Hash>(h: H, s: &mut dyn Hasher) {
    h.hash(&mut WrappedHasher(s))
}
