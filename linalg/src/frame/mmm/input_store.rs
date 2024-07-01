use downcast_rs::{impl_downcast, Downcast};
use dyn_clone::DynClone;
use dyn_hash::DynHash;
use std::alloc::Layout;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::sync::Arc;
use tract_data::internal::*;

pub trait MMMInputFormat: Downcast + Debug + DynHash + DynClone + Send + Sync + Display {
    fn can_prepare_types(&self) -> Vec<DatumType>;
    fn prepare_tensor(
        &self,
        t: &Tensor,
        k_axis: usize,
        mn_axis: usize,
    ) -> TractResult<Box<dyn MMMInputValue>>;
    fn r(&self) -> usize;
}
dyn_clone::clone_trait_object!(MMMInputFormat);
impl_downcast!(MMMInputFormat);
dyn_hash::hash_trait_object!(MMMInputFormat);

pub trait MMMInputValue: DynClone + Debug + DynHash + Send + Sync + Display {
    fn scratch_panel_buffer_layout(&self) -> Option<Layout>;
    fn panel_bytes(&self, i: usize, buffer: Option<*mut u8>) -> TractResult<*const u8>;
    fn panels_count(&self) -> usize {
        self.mn().divceil(self.r())
    }
    fn mn(&self) -> usize;
    fn r(&self) -> usize;
    fn k(&self) -> usize;
}
dyn_clone::clone_trait_object!(MMMInputValue);
dyn_hash::hash_trait_object!(MMMInputValue);

impl From<Box<dyn MMMInputValue>> for Opaque {
    fn from(value: Box<dyn MMMInputValue>) -> Self {
        Opaque(Arc::new(value))
    }
}

impl OpaquePayload for Box<dyn MMMInputValue> {}

#[derive(Clone, Hash)]
pub struct EagerPackedInput {
    pub format: Box<dyn MMMInputFormat>,
    pub packed: Blob,
    pub panel_bytes: usize,
    pub mn: usize,
    pub k: usize,
}

impl MMMInputValue for EagerPackedInput {
    fn scratch_panel_buffer_layout(&self) -> Option<Layout> {
        None
    }
    fn panel_bytes(&self, i: usize, _buffer: Option<*mut u8>) -> TractResult<*const u8> {
        unsafe { Ok(self.packed.as_ptr().add(i * self.panel_bytes)) }
    }
    fn k(&self) -> usize {
        self.k
    }
    fn mn(&self) -> usize {
        self.mn
    }
    fn r(&self) -> usize {
        self.format.r()
    }
}

impl Display for EagerPackedInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Eager {} tensor (mn={} k={})", self.format, self.mn(), self.k())
    }
}

impl Debug for EagerPackedInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as Display>::fmt(self, f)
    }
}
