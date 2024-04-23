use dyn_hash::DynHash;
use std::alloc::Layout;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::sync::Arc;
use tract_data::internal::*;

pub trait MMMInput: dyn_clone::DynClone + Debug + DynHash + Send + Sync + Display {
    fn scratch_panel_buffer_layout(&self) -> Option<Layout>;
    fn panel(&self, i: usize, buffer: Option<*mut u8>) -> *const u8;
}
dyn_clone::clone_trait_object!(MMMInput);
dyn_hash::hash_trait_object!(MMMInput);

impl From<Box<dyn MMMInput>> for Opaque {
    fn from(value: Box<dyn MMMInput>) -> Self {
        Opaque(Arc::new(value))
    }
}

impl OpaquePayload for Box<dyn MMMInput> {
}

#[derive(Debug, Clone, Hash)]
pub struct EagerPackedInput {
    pub packed: Tensor,
    pub panel_bytes: usize,
}

impl MMMInput for EagerPackedInput {
    fn scratch_panel_buffer_layout(&self) -> Option<Layout> {
        None
    }
    fn panel(&self, i: usize, _buffer: Option<*mut u8>) -> *const u8 {
        unsafe { self.packed.as_ptr_unchecked::<u8>().add(i * self.panel_bytes) }
    }
}

impl Display for EagerPackedInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Eagerly packed tensor")
    }
}
